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

        stats_calc = StatsCalculator(
            pitch_length_m=40.0,
            pitch_width_m=20.0,
            frame_width=width,
            frame_height=height,
            fps=fps,
        )

        # Map players
        db_players = db.query(Player).filter(Player.match_id == match_id).all()
        player_by_track = {p.track_id: p for p in db_players}

        # Clear existing stats for idempotency
        db.query(PlayerStat).filter(PlayerStat.match_id == match_id).delete()

        # Compute each player's stats
        home_distance = 0.0
        away_distance = 0.0

        for track_id, p_track in tracked_players.items():
            p_record = player_by_track.get(track_id)
            if not p_record:
                continue

            metrics = stats_calc.compute_player_stats(p_track, events, total_frames)

            stat_entry = PlayerStat(
                match_id=match_id,
                player_id=p_record.id,
                distance_m=metrics["distance_m"],
                top_speed_kmh=metrics["top_speed_kmh"],
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

        duration_sec = int(total_frames / fps) if fps > 0 else 0

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
