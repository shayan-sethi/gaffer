import logging
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Match, MatchStatus, Player, PlayerStat, Event
from app.cv.stats import StatsCalculator
from app.cv.tracker import TrackedObject
from app.cv.events import DetectedEvent
from app.cv.calibration import PitchCalibration

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.stats.aggregate_stats_task")
def aggregate_stats_task(self, event_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes spatial heatmaps, physical distance, top speed, and possession stats
    for all players, and marks match as COMPLETED.
    """
    match_id = event_payload["match_id"]
    total_frames = event_payload.get("total_frames", 150)
    fps = event_payload.get("fps", 30.0)
    width = event_payload.get("frame_width", 1280)
    height = event_payload.get("frame_height", 720)
    pitch_bounds = event_payload.get("pitch_bounds")
    raw_tracks = event_payload.get("tracks", {})
    raw_events = event_payload.get("events", [])

    logger.info(f"Starting aggregate_stats_task for match_id={match_id}...")
    db: Session = SessionLocal()

    try:
        match = db.query(Match).filter(Match.id == match_id).first()
        if not match:
            raise ValueError(f"Match {match_id} not found.")

        # Reconstruct events
        events: List[DetectedEvent] = [
            DetectedEvent(
                frame_no=e["frame_no"],
                timestamp_ms=e["timestamp_ms"],
                event_type=e["event_type"],
                player_track_id=e.get("player_track_id"),
                team=e.get("team"),
                data_json=e.get("data_json", {}),
            )
            for e in raw_events
        ]

        # Reconstruct player TrackedObjects
        tracked_players: Dict[int, TrackedObject] = {}
        for t_id_str, t_data in raw_tracks.items():
            if t_data["class_name"] != "player":
                continue
            t_id = int(t_id_str)
            obj = TrackedObject(
                track_id=t_id,
                class_name="player",
                bbox=[0, 0, 0, 0],
                centroid=tuple(t_data["history"][-1][1]) if t_data["history"] else (0, 0),
                frame_no=0,
            )
            obj.team = t_data["team"]
            obj.history = [(f, tuple(pt)) for f, pt in t_data["history"]]
            tracked_players[t_id] = obj

        # Scale meters-per-pixel off the actual detected pitch region, not the raw
        # frame - the frame includes crowd/stands/sky margins outside the pitch,
        # so using frame_width/frame_height directly overstated the real pixel
        # extent of the 40m/20m pitch and skewed every distance/speed/sprint number.
        # (frame_width/frame_height are kept as-is for the heatmap grid, which
        # buckets raw-frame-space positions and must normalize against those.)
        if pitch_bounds:
            px1, py1, px2, py2 = pitch_bounds
            pitch_px_width = max(1, px2 - px1)
            pitch_px_height = max(1, py2 - py1)
        else:
            pitch_px_width = None
            pitch_px_height = None

        calibration = PitchCalibration.from_points_json(
            event_payload.get("calibration_points"), pitch_length_m=40.0, pitch_width_m=20.0
        )

        stats_calc = StatsCalculator(
            pitch_length_m=40.0,
            pitch_width_m=20.0,
            frame_width=width,
            frame_height=height,
            fps=fps,
            pitch_px_width=pitch_px_width,
            pitch_px_height=pitch_px_height,
            calibration=calibration,
        )

        # Map players
        db_players = db.query(Player).filter(Player.match_id == match_id).all()
        player_by_track = {p.track_id: p for p in db_players}

        # Clear existing stats for idempotency
        db.query(PlayerStat).filter(PlayerStat.match_id == match_id).delete()

        # Possession-frame counts (real time-in-possession, not a pass-count proxy -
        # see EventDetector.possession_frames / player_possession_frames). Keys may
        # come back as strings after a JSON round-trip through the Celery broker,
        # so normalize player track-id keys to int here same as tracks/events above.
        possession_frames = event_payload.get("possession_frames", {"HOME": 0, "AWAY": 0})
        player_possession_frames = {
            int(k): v for k, v in event_payload.get("player_possession_frames", {}).items()
        }
        total_possession_frames = sum(possession_frames.values())

        # Compute each player's stats
        home_distance = 0.0
        away_distance = 0.0

        for track_id, p_track in tracked_players.items():
            p_record = player_by_track.get(track_id)
            if not p_record:
                continue

            metrics = stats_calc.compute_player_stats(
                p_track, events, total_frames,
                possession_frames_by_player=player_possession_frames,
                total_possession_frames=total_possession_frames,
            )

            stat_entry = PlayerStat(
                match_id=match_id,
                player_id=p_record.id,
                distance_m=metrics["distance_m"],
                top_speed_kmh=metrics["top_speed_kmh"],
                avg_speed_kmh=metrics["avg_speed_kmh"],
                sprints=metrics["sprints"],
                passes=metrics["passes"],
                shots=metrics["shots"],
                possession_pct=metrics["possession_pct"],
                heatmap_json=metrics["heatmap_json"],
            )
            db.add(stat_entry)

            if p_track.team == "HOME":
                home_distance += metrics["distance_m"]
            else:
                away_distance += metrics["distance_m"]

        # Compute match scores from GOAL events
        home_goals = sum(1 for e in events if e.event_type == "GOAL" and e.team == "HOME")
        away_goals = sum(1 for e in events if e.event_type == "GOAL" and e.team == "AWAY")

        home_xg = round(sum(e.data_json.get("xg", 0.0) for e in events if e.event_type == "SHOT" and e.team == "HOME"), 3)
        away_xg = round(sum(e.data_json.get("xg", 0.0) for e in events if e.event_type == "SHOT" and e.team == "AWAY"), 3)
        home_shots = sum(1 for e in events if e.event_type == "SHOT" and e.team == "HOME")
        away_shots = sum(1 for e in events if e.event_type == "SHOT" and e.team == "AWAY")
        home_corners = sum(1 for e in events if e.event_type == "CORNER" and e.team == "HOME")
        away_corners = sum(1 for e in events if e.event_type == "CORNER" and e.team == "AWAY")

        duration_sec = int(total_frames / fps) if fps > 0 else 0

        if total_possession_frames > 0:
            home_possession_pct = round(100 * possession_frames.get("HOME", 0) / total_possession_frames, 1)
        else:
            home_possession_pct = 50.0
        away_possession_pct = round(100.0 - home_possession_pct, 1)

        # Update Match record
        match.status = MatchStatus.COMPLETED
        match.duration_sec = duration_sec
        match.end_time = datetime.utcnow()
        match.meta_data = {
            "score": {"home": home_goals, "away": away_goals},
            "total_distance_m": round(home_distance + away_distance, 2),
            "total_goals": home_goals + away_goals,
            "total_passes": sum(1 for e in events if e.event_type == "PASS"),
            "total_events": len(events),
            "zone_touches": event_payload.get("zone_touches"),
            "possession_pct": {"home": home_possession_pct, "away": away_possession_pct},
            "xg": {"home": home_xg, "away": away_xg},
            "shots": {"home": home_shots, "away": away_shots},
            "corners": {"home": home_corners, "away": away_corners},
            "processed_at": datetime.utcnow().isoformat(),
        }

        db.commit()
        logger.info(f"Match {match_id} analysis successfully completed! Final Score: HOME {home_goals} - {away_goals} AWAY")

        return {
            "match_id": match_id,
            "status": "COMPLETED",
            "score": {"home": home_goals, "away": away_goals},
            "duration_sec": duration_sec,
        }

    except Exception as e:
        logger.error(f"Error in aggregate_stats_task for match {match_id}: {e}", exc_info=True)
        if match:
            match.status = MatchStatus.FAILED
            match.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()
