import os
import logging
import cv2
import numpy as np
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Match, MatchStatus, Player, Detection
from app.cv.detector import YOLOv8Detector
from app.cv.tracker import CentroidTracker
from app.tasks.ingest import download_match_video
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.track.track_players_task")
def track_players_task(self, match_id: str, s3_url: str = None) -> Dict[str, Any]:
    """
    Downloads match video, executes YOLOv8 detection and Centroid tracking across all frames,
    and saves player and detection records to PostgreSQL.
    """
    logger.info(f"Starting track_players_task for match_id={match_id}, s3_url={s3_url}")
    db: Session = SessionLocal()

    try:
        match = db.query(Match).filter(Match.id == match_id).first()
        if not match:
            raise ValueError(f"Match with ID {match_id} not found.")

        match.status = MatchStatus.PROCESSING
        db.commit()

        video_url = s3_url or match.video_s3_url or "synthetic"
        local_video_path = os.path.join(settings.STORAGE_LOCAL_DIR, f"{match_id}.mp4")

        # 1. Download/prepare video
        video_path = download_match_video(video_url, local_video_path)

        # 2. Open video stream
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 150

        detector = YOLOv8Detector(
            model_path=settings.YOLO_MODEL_PATH,
            conf_threshold=settings.CONFIDENCE_THRESHOLD,
            iou_threshold=settings.IOU_THRESHOLD,
        )
        tracker = CentroidTracker(max_lost_frames=30, max_distance_px=90.0)

        frame_no = 0
        all_detections_batch: List[Detection] = []
        sampled_rate = 2  # Process every 2nd frame for speed/efficiency while maintaining tracks

        logger.info(f"Processing {total_frames} frames for match_id={match_id} (fps={fps}, res={width}x{height})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if frame_no % sampled_rate != 0 and frame_no != total_frames:
                continue

            timestamp_ms = int((frame_no / fps) * 1000)

            # Detect objects (players and ball)
            detections = detector.detect_frame(frame)

            # Update tracker
            active_tracks = tracker.update(detections, frame_no)

            # Build Detection ORM records
            for d in detections:
                matched_track = next(
                    (t for t in active_tracks if t.class_name == d.class_name and np.allclose(t.bbox, d.bbox, atol=20)),
                    None
                )
                track_id = matched_track.track_id if matched_track else None

                all_detections_batch.append(
                    Detection(
                        match_id=match_id,
                        frame_no=frame_no,
                        timestamp_ms=timestamp_ms,
                        track_id=track_id,
                        class_name=d.class_name,
                        bbox=d.bbox,
                        confidence=d.confidence,
                    )
                )

        cap.release()

        # Batch insert detections
        if all_detections_batch:
            db.bulk_save_objects(all_detections_batch)
            db.commit()

        # Create Player records for all player tracks found
        player_tracks = {t_id: t for t_id, t in tracker.tracks.items() if t.class_name == "player"}
        created_players: Dict[int, str] = {}  # track_id -> player_id (UUID)

        for track_id, t in player_tracks.items():
            existing_player = db.query(Player).filter(
                Player.match_id == match_id,
                Player.track_id == track_id
            ).first()

            if not existing_player:
                p = Player(
                    match_id=match_id,
                    track_id=track_id,
                    jersey_number=track_id,
                    player_name=f"Player #{track_id}",
                    team=t.team,
                )
                db.add(p)
                db.flush()
                created_players[track_id] = p.id
            else:
                created_players[track_id] = existing_player.id

        db.commit()

        # Prepare serializable summary for downstream tasks
        serializable_tracks = {}
        for t_id, t in tracker.tracks.items():
            serializable_tracks[str(t_id)] = {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "team": t.team,
                "history": [(f, [round(pt[0], 2), round(pt[1], 2)]) for f, pt in t.history],
                "player_id": created_players.get(t.track_id),
            }

        logger.info(f"Tracking complete for match_id={match_id}. Found {len(player_tracks)} players across {frame_no} frames.")

        return {
            "match_id": match_id,
            "total_frames": frame_no,
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "tracks": serializable_tracks,
        }

    except Exception as e:
        logger.error(f"Error in track_players_task for match {match_id}: {e}", exc_info=True)
        if match:
            match.status = MatchStatus.FAILED
            match.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()
