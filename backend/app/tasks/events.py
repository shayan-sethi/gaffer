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
    home_attacks_right: bool = True,
    pitch_bounds_history: Optional[List[Tuple[int, Tuple[int, int, int, int]]]] = None,
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

    A corner is awarded to the ATTACKING team when the ball goes out behind
    the DEFENDING team's own goal line - so the team credited with the corner
    is whoever attacks the goal line the ball just went out behind, per
    `home_attacks_right` (a per-match camera/kickoff convention, not something
    derivable from pitch geometry alone - see EventDetector.home_attacks_right).

    `pitch_bounds_history`, when given, is a (frame_no, bbox) series - the same
    live-tracked pitch region recomputed every processed frame during tracking
    (see track_players_task) - and the pixel-fraction fallback below looks up
    the bbox nearest each reappearance frame instead of one fixed `pitch_bounds`
    for the whole match. On a panning camera, a single static box drifts wrong
    as play moves around the pitch, exactly the way it did for goal detection
    (see EventDetector.update_pitch_bounds) - a corner check anchored to a stale
    box can misjudge which physical side of the pitch a reappearance is on.
    """
    history_frames = [f for f, _ in pitch_bounds_history] if pitch_bounds_history else []

    def _bounds_at(frame_no: int) -> Tuple[int, int, int, int]:
        if history_frames:
            idx = min(bisect.bisect_right(history_frames, frame_no), len(history_frames)) - 1
            if idx >= 0:
                return tuple(pitch_bounds_history[idx][1])
        if pitch_bounds is not None:
            return tuple(pitch_bounds)
        return (0, 0, frame_width, frame_height)

    if calibration is not None:
        # A real, fixed-radius corner region in meters - unlike the pixel-
        # fraction fallback below, this is the same real size regardless of
        # whether that corner sits near the camera (large in pixels) or far
        # from it (small in pixels), the same perspective distortion
        # PitchCalibration corrects for distance/speed.
        right_side_team = "HOME" if home_attacks_right else "AWAY"
        left_side_team = "AWAY" if home_attacks_right else "HOME"
        world_corners = {
            left_side_team: [(0.0, 0.0), (0.0, pitch_width_m)],
            right_side_team: [(pitch_length_m, 0.0), (pitch_length_m, pitch_width_m)],
        }

        def attacking_team_for_corner(x: float, y: float, frame_no: int) -> Optional[str]:
            wx, wy = calibration.pixel_to_world_m(x, y)
            for team, corners in world_corners.items():
                for cx, cy in corners:
                    if math.hypot(wx - cx, wy - cy) <= corner_radius_m:
                        return team
            return None
    else:
        def attacking_team_for_corner(x: float, y: float, frame_no: int) -> Optional[str]:
            px1, py1, px2, py2 = _bounds_at(frame_no)
            pitch_w = max(1, px2 - px1)
            pitch_h = max(1, py2 - py1)
            x_margin = pitch_w * 0.08
            y_margin = pitch_h * 0.12
            near_left, near_right = x <= px1 + x_margin, x >= px2 - x_margin
            near_top, near_bottom = y <= py1 + y_margin, y >= py2 - y_margin
            if (near_left or near_right) and (near_top or near_bottom):
                # Same left/right attacking convention as GOAL/zone checks (see
                # home_attacks_right) - a corner won on a side belongs to
                # whichever team attacks that side.
                if near_right:
                    return "HOME" if home_attacks_right else "AWAY"
                return "AWAY" if home_attacks_right else "HOME"
            return None

    events: List[DetectedEvent] = []
    last_corner_frame = -int(fps * 8)
    for i in range(1, len(ball_sorted_frames)):
        f_before, f_after = ball_sorted_frames[i - 1], ball_sorted_frames[i]
        gap = f_after - f_before
        if gap <= max_gap_ball:
            continue  # ordinary sampling gap, not a real stoppage

        x, y = ball_positions[f_after]
        team = attacking_team_for_corner(x, y, f_after)
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


def _detect_throw_ins(
    ball_sorted_frames: List[int],
    ball_positions: Dict[int, Tuple[float, float]],
    pitch_bounds: Optional[Tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    fps: float,
    max_gap_ball: int,
    possessor_team_by_frame: Dict[int, Optional[str]],
    calibration: Optional[PitchCalibration] = None,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    sideline_radius_m: float = 3.0,
    pitch_bounds_history: Optional[List[Tuple[int, Tuple[int, int, int, int]]]] = None,
) -> List[DetectedEvent]:
    """
    Approximate throw-in detection - same real-stoppage signal _detect_corners
    uses (the ball goes genuinely undetected, then reappears), but classified
    by reappearing along a TOUCHLINE away from either goal-line end, rather
    than right at a corner. A throw-in goes to whichever team did NOT put the
    ball out, so - unlike a corner, which is about which end of the pitch the
    ball went out behind - team attribution here comes from who last had
    genuine possession (see possessor_team_by_frame, sampled during
    EventDetector's own frame loop) immediately before the gap, crediting the
    OTHER team.
    """
    history_frames = [f for f, _ in pitch_bounds_history] if pitch_bounds_history else []

    def _bounds_at(frame_no: int) -> Tuple[int, int, int, int]:
        if history_frames:
            idx = min(bisect.bisect_right(history_frames, frame_no), len(history_frames)) - 1
            if idx >= 0:
                return tuple(pitch_bounds_history[idx][1])
        if pitch_bounds is not None:
            return tuple(pitch_bounds)
        return (0, 0, frame_width, frame_height)

    if calibration is not None:
        def is_touchline_not_corner(x: float, y: float, frame_no: int) -> bool:
            wx, wy = calibration.pixel_to_world_m(x, y)
            near_touchline = wy <= sideline_radius_m or wy >= pitch_width_m - sideline_radius_m
            near_goal_line = wx <= sideline_radius_m or wx >= pitch_length_m - sideline_radius_m
            return near_touchline and not near_goal_line
    else:
        def is_touchline_not_corner(x: float, y: float, frame_no: int) -> bool:
            px1, py1, px2, py2 = _bounds_at(frame_no)
            pitch_w = max(1, px2 - px1)
            pitch_h = max(1, py2 - py1)
            x_margin = pitch_w * 0.08
            y_margin = pitch_h * 0.12
            near_top, near_bottom = y <= py1 + y_margin, y >= py2 - y_margin
            near_left, near_right = x <= px1 + x_margin, x >= px2 - x_margin
            return (near_top or near_bottom) and not (near_left or near_right)

    def _last_possessor_before(frame_no: int) -> Optional[str]:
        # Nearest recorded possessor at or before this frame - the ball was
        # already gone (that's the whole premise of a stoppage), so there's
        # rarely an exact entry for frame_no itself.
        for f in range(frame_no, max(0, frame_no - int(fps * 5)), -1):
            team = possessor_team_by_frame.get(f)
            if team is not None:
                return team
        return None

    events: List[DetectedEvent] = []
    last_throw_in_frame = -int(fps * 8)
    for i in range(1, len(ball_sorted_frames)):
        f_before, f_after = ball_sorted_frames[i - 1], ball_sorted_frames[i]
        gap = f_after - f_before
        if gap <= max_gap_ball:
            continue  # ordinary sampling gap, not a real stoppage

        x, y = ball_positions[f_after]
        if not is_touchline_not_corner(x, y, f_after):
            continue
        if (f_after - last_throw_in_frame) < fps * 8:
            continue

        team_that_lost_it = _last_possessor_before(f_before)
        if team_that_lost_it not in ("HOME", "AWAY"):
            continue  # can't credit a team we don't know we're crediting
        team = "AWAY" if team_that_lost_it == "HOME" else "HOME"

        events.append(DetectedEvent(
            frame_no=f_after,
            timestamp_ms=int((f_after / fps) * 1000),
            event_type="THROW_IN",
            team=team,
            data_json={"team": team, "gap_frames": gap, "confidence": 0.5},
        ))
        last_throw_in_frame = f_after

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
    # (frame_no, bbox) samples of the SAME live-tracked pitch region computed
    # every processed frame during tracking (see track_players_task), used to
    # keep goal/corner zones tracking the camera's pan instead of freezing on
    # whichever single frame's segmentation happened to be the last one saved
    # (see EventDetector.update_pitch_bounds). Older cached payloads won't
    # have this key - falls back to the single static `pitch_bounds` then.
    pitch_bounds_history = tracking_payload.get("pitch_bounds_history") or []
    pbh_frames = [f for f, _ in pitch_bounds_history]
    # Which goal each team attacks - a per-match camera/kickoff convention
    # (see EventDetector.home_attacks_right), not derivable from the footage
    # itself, so it's carried through the payload rather than assumed.
    home_attacks_right = tracking_payload.get("home_attacks_right", True)
    # Which pixel axis the goals sit along - "x" for a sideline camera
    # (default, goals at left/right), "y" for a camera mounted behind/above
    # one goal shooting down the pitch length (see EventDetector.goal_axis).
    goal_axis = tracking_payload.get("goal_axis", "x")
    # Real pitch dimensions - default to a 5-a-side turf pitch (what this
    # pipeline was built against), but a full-size 11-a-side pitch needs the
    # real ~105x68m so distance/speed/xG and the corner/throw-in radius checks
    # aren't scaled against the wrong pitch entirely.
    pitch_length_m = tracking_payload.get("pitch_length_m", 40.0)
    pitch_width_m = tracking_payload.get("pitch_width_m", 20.0)
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
            tracking_payload.get("calibration_points"), pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m
        )

        # Initialize detector
        event_detector = EventDetector(
            frame_width=width,
            frame_height=height,
            fps=fps,
            pitch_bounds=tuple(pitch_bounds) if pitch_bounds else None,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            calibration=calibration,
            home_attacks_right=home_attacks_right,
            goal_axis=goal_axis,
        )

        # Simulate frame-by-frame processing for events
        detected_events_orm: List[Event] = []
        # Sampled every frame during the main loop below so _detect_throw_ins
        # can look up "who had the ball right before it went out" after the
        # fact - EventDetector doesn't persist this itself once a frame passes.
        possessor_team_by_frame: Dict[int, Optional[str]] = {}

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

            if pbh_frames:
                idx = min(bisect.bisect_right(pbh_frames, frame_no), len(pbh_frames)) - 1
                if idx >= 0:
                    event_detector.update_pitch_bounds(tuple(pitch_bounds_history[idx][1]))

            frame_evts = event_detector.process_frame(frame_tracks, frame_no)
            possessor_team_by_frame[frame_no] = event_detector.last_possessor_team

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
        throw_in_events: List[DetectedEvent] = []
        if ball_track_id is not None:
            corner_events = _detect_corners(
                track_frames_sorted[ball_track_id],
                track_positions[ball_track_id],
                tuple(pitch_bounds) if pitch_bounds else None,
                width, height, fps, max_gap_ball,
                calibration=calibration,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                home_attacks_right=home_attacks_right,
                pitch_bounds_history=pitch_bounds_history or None,
            )
            throw_in_events = _detect_throw_ins(
                track_frames_sorted[ball_track_id],
                track_positions[ball_track_id],
                tuple(pitch_bounds) if pitch_bounds else None,
                width, height, fps, max_gap_ball,
                possessor_team_by_frame,
                calibration=calibration,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                pitch_bounds_history=pitch_bounds_history or None,
            )
            for e in corner_events + throw_in_events:
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
            f"Saved {len(detected_events_orm)} events "
            f"({len(corner_events)} corners, {len(throw_in_events)} throw-ins)."
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
