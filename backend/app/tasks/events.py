import bisect
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Match, Event, Player
from app.cv.events import EventDetector, DetectedEvent
from app.cv.tracker import TrackedObject
from app.cv.calibration import PitchCalibration

logger = logging.getLogger(__name__)


def _position_at_frame(
    sorted_frames: List[int],
    positions_by_frame: Dict[int, Tuple[float, float]],
    frame_no: int,
    max_gap_frames: int,
) -> Optional[Tuple[float, float]]:
    """
    Real position if this track was detected exactly on `frame_no`, otherwise a
    linear interpolation between its nearest surrounding detections - as long as
    the ball's own real detection rate is well under 100%, treating "not detected
    this exact sampled frame" as "not present" silently drops most true touches/
    passes. Interpolating between two real, already-detected points of the same
    continuously-tracked object isn't fabrication, it's filling in motion between
    known samples - the same idea as video frame interpolation. Bounded by
    `max_gap_frames` so a genuinely long gap (ball out of play, half-time) isn't
    bridged into a fake continuous trajectory.
    """
    if not sorted_frames:
        return None
    if frame_no in positions_by_frame:
        return positions_by_frame[frame_no]

    idx = bisect.bisect_left(sorted_frames, frame_no)
    if idx == 0 or idx == len(sorted_frames):
        return None  # before the first or after the last known sighting

    f_before, f_after = sorted_frames[idx - 1], sorted_frames[idx]
    if (f_after - f_before) > max_gap_frames:
        return None

    t = (frame_no - f_before) / (f_after - f_before)
    x1, y1 = positions_by_frame[f_before]
    x2, y2 = positions_by_frame[f_after]
    return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)


def _detect_corners(
    ball_sorted_frames: List[int],
    ball_positions: Dict[int, Tuple[float, float]],
    pitch_bounds: Optional[Tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    fps: float,
    max_gap_ball: int,
    calibration: Optional[PitchCalibration] = None,
    pitch_length_m: float = 40.0,
    pitch_width_m: float = 20.0,
    corner_radius_m: float = 3.0,
) -> List[DetectedEvent]:
    """
    Approximate corner-kick detection: the ball is genuinely undetected for a
    real stoppage (a gap wider than the interpolation window above, i.e. not
    just a normal miss between samples), then reappears right in one of the
    four pitch corners - the one restart position that's specific to a corner
    kick (a goal-kick restarts more centrally near the box, a throw-in restarts
    anywhere along the touchline, not the corner specifically). This is a real,
    if approximate, geometric signal - not a fabricated stat - but it can't
    distinguish a corner from, say, a throw-in that happens to land near the
    corner flag, so treat it as a rough estimate.
    """
    if pitch_bounds is not None:
        px1, py1, px2, py2 = pitch_bounds
    else:
        px1, py1, px2, py2 = 0, 0, frame_width, frame_height

    if calibration is not None:
        # A real, fixed-radius corner region in meters - unlike the pixel-
        # fraction fallback below, this is the same real size regardless of
        # whether that corner sits near the camera (large in pixels) or far
        # from it (small in pixels), the same perspective distortion
        # PitchCalibration corrects for distance/speed.
        world_corners = {
            "AWAY": [(0.0, 0.0), (0.0, pitch_width_m)],   # AWAY attacks LEFT (low x)
            "HOME": [(pitch_length_m, 0.0), (pitch_length_m, pitch_width_m)],  # HOME attacks RIGHT
        }

        def attacking_team_for_corner(x: float, y: float) -> Optional[str]:
            wx, wy = calibration.pixel_to_world_m(x, y)
            for team, corners in world_corners.items():
                for cx, cy in corners:
                    if math.hypot(wx - cx, wy - cy) <= corner_radius_m:
                        return team
            return None
    else:
        pitch_w = max(1, px2 - px1)
        pitch_h = max(1, py2 - py1)
        x_margin = pitch_w * 0.08
        y_margin = pitch_h * 0.12

        def attacking_team_for_corner(x: float, y: float) -> Optional[str]:
            near_left, near_right = x <= px1 + x_margin, x >= px2 - x_margin
            near_top, near_bottom = y <= py1 + y_margin, y >= py2 - y_margin
            if (near_left or near_right) and (near_top or near_bottom):
                # Same left/right attacking convention as GOAL/zone checks: HOME
                # attacks the right side, so a corner won there is HOME's.
                return "HOME" if near_right else "AWAY"
            return None

    events: List[DetectedEvent] = []
    last_corner_frame = -int(fps * 8)
    for i in range(1, len(ball_sorted_frames)):
        f_before, f_after = ball_sorted_frames[i - 1], ball_sorted_frames[i]
        gap = f_after - f_before
        if gap <= max_gap_ball:
            continue  # ordinary sampling gap, not a real stoppage

        x, y = ball_positions[f_after]
        team = attacking_team_for_corner(x, y)
        if team is None or (f_after - last_corner_frame) < fps * 8:
            continue

        events.append(DetectedEvent(
            frame_no=f_after,
            timestamp_ms=int((f_after / fps) * 1000),
            event_type="CORNER",
            team=team,
            data_json={"attacking_team": team, "gap_frames": gap, "confidence": 0.5},
        ))
        last_corner_frame = f_after

    return events


@celery_app.task(bind=True, name="app.tasks.events.detect_events_task")
def detect_events_task(self, tracking_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes player and ball tracking trajectories to detect goals, passes, shots, and possession shifts.
    Saves event records to PostgreSQL.
    """
    match_id = tracking_payload["match_id"]
    total_frames = tracking_payload.get("total_frames", 150)
    fps = tracking_payload.get("fps", 30.0)
    width = tracking_payload.get("frame_width", 1280)
    height = tracking_payload.get("frame_height", 720)
    pitch_bounds = tracking_payload.get("pitch_bounds")
    raw_tracks = tracking_payload.get("tracks", {})

    logger.info(f"Starting detect_events_task for match_id={match_id} ({total_frames} frames)...")
    db: Session = SessionLocal()

    try:
        # Reconstruct TrackedObjects from serializable payload, plus a sorted-frame
        # index per track for O(log n) exact/interpolated lookups instead of an
        # O(history length) scan per frame (this loop runs total_frames times).
        tracked_objects: Dict[int, TrackedObject] = {}
        track_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}
        track_frames_sorted: Dict[int, List[int]] = {}
        for t_id_str, t_data in raw_tracks.items():
            t_id = int(t_id_str)
            obj = TrackedObject(
                track_id=t_id,
                class_name=t_data["class_name"],
                bbox=[0, 0, 0, 0],
                centroid=tuple(t_data["history"][-1][1]) if t_data["history"] else (0, 0),
                frame_no=0,
            )
            obj.team = t_data["team"]
            obj.history = [(f, tuple(pt)) for f, pt in t_data["history"]]
            tracked_objects[t_id] = obj
            positions_by_frame = {f: pt for f, pt in obj.history}
            track_positions[t_id] = positions_by_frame
            track_frames_sorted[t_id] = sorted(positions_by_frame.keys())

        # Ball detections are far sparser than player detections (broadcast-distance
        # footage), so give the ball a generous interpolation window - a few seconds
        # is still clearly "the same continuous touch/dribble/pass", not a real gap.
        # Players are detected reliably enough that a much tighter window is plenty
        # and avoids interpolating through a genuine long absence (sub, off-frame).
        max_gap_ball = int(fps * 3)
        max_gap_player = int(fps * 1)

        # Map track_id -> DB Player record
        player_map = {
            p.track_id: p
            for p in db.query(Player).filter(Player.match_id == match_id).all()
        }

        calibration = PitchCalibration.from_points_json(
            tracking_payload.get("calibration_points"), pitch_length_m=40.0, pitch_width_m=20.0
        )

        # Initialize detector
        event_detector = EventDetector(
            frame_width=width,
            frame_height=height,
            fps=fps,
            pitch_bounds=tuple(pitch_bounds) if pitch_bounds else None,
            calibration=calibration,
        )

        # Simulate frame-by-frame processing for events
        detected_events_orm: List[Event] = []

        for frame_no in range(1, total_frames + 1):
            # Find active (real or interpolated) tracks at this frame
            frame_tracks = []
            for t_id, obj in tracked_objects.items():
                max_gap = max_gap_ball if obj.class_name == "ball" else max_gap_player
                pt = _position_at_frame(track_frames_sorted[t_id], track_positions[t_id], frame_no, max_gap)
                if pt is None:
                    continue
                current_t = TrackedObject(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    bbox=[pt[0] - 20, pt[1] - 40, pt[0] + 20, pt[1]],
                    centroid=pt,
                    frame_no=frame_no,
                )
                current_t.team = obj.team
                frame_tracks.append(current_t)

            frame_evts = event_detector.process_frame(frame_tracks, frame_no)

            for e in frame_evts:
                player_db_id = player_map[e.player_track_id].id if e.player_track_id in player_map else None
                orm_event = Event(
                    match_id=match_id,
                    frame_no=e.frame_no,
                    timestamp_ms=e.timestamp_ms,
                    event_type=e.event_type,
                    player_id=player_db_id,
                    team=e.team,
                    data_json=e.data_json,
                )
                detected_events_orm.append(orm_event)

        # Corner detection needs the ball's raw (non-interpolated) gap structure,
        # which lives outside EventDetector's per-frame view - computed as a
        # separate pass over the ball track's real detection history.
        ball_track_id = next((t_id for t_id, obj in tracked_objects.items() if obj.class_name == "ball"), None)
        corner_events: List[DetectedEvent] = []
        if ball_track_id is not None:
            corner_events = _detect_corners(
                track_frames_sorted[ball_track_id],
                track_positions[ball_track_id],
                tuple(pitch_bounds) if pitch_bounds else None,
                width, height, fps, max_gap_ball,
                calibration=calibration,
            )
            for e in corner_events:
                detected_events_orm.append(Event(
                    match_id=match_id,
                    frame_no=e.frame_no,
                    timestamp_ms=e.timestamp_ms,
                    event_type=e.event_type,
                    player_id=None,
                    team=e.team,
                    data_json=e.data_json,
                ))

        if detected_events_orm:
            db.bulk_save_objects(detected_events_orm)
            db.commit()

        logger.info(
            f"Event detection complete for match {match_id}. "
            f"Saved {len(detected_events_orm)} events ({len(corner_events)} corners)."
        )

        # Serialize detected events for stats task
        serialized_events = [
            {
                "frame_no": e.frame_no,
                "timestamp_ms": e.timestamp_ms,
                "event_type": e.event_type,
                "player_track_id": e.player_track_id,
                "team": e.team,
                "data_json": e.data_json,
            }
            for e in (event_detector.events + corner_events)
        ]

        return {
            **tracking_payload,
            "events": serialized_events,
            "zone_touches": event_detector.zone_touches,
            "possession_frames": event_detector.possession_frames,
            "player_possession_frames": event_detector.player_possession_frames,
        }

    except Exception as e:
        logger.error(f"Error in detect_events_task for match {match_id}: {e}", exc_info=True)
        raise
    finally:
        db.close()
