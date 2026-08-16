import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Match, Event, Player
from app.cv.events import EventDetector
from app.cv.tracker import TrackedObject

logger = logging.getLogger(__name__)


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
    raw_tracks = tracking_payload.get("tracks", {})

    logger.info(f"Starting detect_events_task for match_id={match_id} ({total_frames} frames)...")
    db: Session = SessionLocal()

    try:
        # Reconstruct TrackedObjects from serializable payload
        tracked_objects: Dict[int, TrackedObject] = {}
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

        # Map track_id -> DB Player record
        player_map = {
            p.track_id: p
            for p in db.query(Player).filter(Player.match_id == match_id).all()
        }

        # Initialize detector
        event_detector = EventDetector(frame_width=width, frame_height=height, fps=fps)

        # Simulate frame-by-frame processing for events
        detected_events_orm: List[Event] = []

        for frame_no in range(1, total_frames + 1):
            # Find active tracks at this frame
            frame_tracks = []
            for t_id, obj in tracked_objects.items():
                for f, pt in obj.history:
                    if f == frame_no:
                        # Temporary object for this frame
                        current_t = TrackedObject(
                            track_id=obj.track_id,
                            class_name=obj.class_name,
                            bbox=[pt[0] - 20, pt[1] - 40, pt[0] + 20, pt[1]],
                            centroid=pt,
                            frame_no=frame_no,
                        )
                        current_t.team = obj.team
                        frame_tracks.append(current_t)
                        break

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

        if detected_events_orm:
            db.bulk_save_objects(detected_events_orm)
            db.commit()

        logger.info(f"Event detection complete for match {match_id}. Saved {len(detected_events_orm)} events.")

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
            for e in event_detector.events
        ]

        return {
            **tracking_payload,
            "events": serialized_events,
        }

    except Exception as e:
        logger.error(f"Error in detect_events_task for match {match_id}: {e}", exc_info=True)
        raise
    finally:
        db.close()
