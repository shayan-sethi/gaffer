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
from app.cv.pitch_mask import PitchMask
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
        is_synthetic_source = video_url == "synthetic" or video_url.startswith("mock://")
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
            allow_fallback_on_empty=is_synthetic_source,
            img_size=settings.YOLO_IMG_SIZE,
        )
        tracker = CentroidTracker(max_lost_frames=45, max_distance_px=90.0)

        frame_no = 0
        all_detections_batch: List[Detection] = []
        sampled_rate = 2  # Process every 2nd frame for speed/efficiency while maintaining tracks
        pitch_mask: PitchMask = None  # Computed lazily from the first frame (camera is fixed all match)

        max_limit = int(os.getenv("MAX_FRAMES", "0"))
        effective_total = min(total_frames, max_limit) if max_limit > 0 else total_frames
        logger.info(f"Processing {effective_total} frames for match_id={match_id} (fps={fps}, res={width}x{height}, GPU={detector.device})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if max_limit > 0 and frame_no > max_limit:
                break

            if frame_no % sampled_rate != 0 and frame_no != effective_total:
                continue

            if frame_no % 300 == 0 or frame_no == effective_total:
                pct = (frame_no / effective_total) * 100
                logger.info(f"CV Tracking Progress: Frame {frame_no}/{effective_total} ({pct:.1f}%) | Active Player Tracks: {len([t for t in tracker.tracks.values() if t.class_name == 'player'])}")

            timestamp_ms = int((frame_no / fps) * 1000)

            if pitch_mask is None or pitch_mask.mask is None:
                # Keep retrying (e.g. past intro/title cards) until segmentation succeeds, then lock it in
                pitch_mask = PitchMask(frame)

            # Detect objects (players and ball)
            detections = detector.detect_frame(frame)

            # Drop "player" detections that fall outside the field of play - sideline
            # spectators, benches, and bystanders are detected by YOLO just as readily
            # as players, and otherwise fragment/inflate track IDs.
            detections = [
                d for d in detections
                if d.class_name != "player" or pitch_mask.contains_point((d.bbox[0] + d.bbox[2]) / 2.0, d.bbox[3])
            ]

            # A single full-frame pass essentially never finds the ball on real broadcast-
            # distance footage - it's too small/low-contrast. Tiled ball-only detection
            # trades extra inference passes for enough effective resolution to see it.
            # Skip this on the synthetic demo canvas: the fallback already places a
            # simulated ball there, and there's no real ball to tile-search for.
            if not is_synthetic_source:
                detections = [d for d in detections if d.class_name != "ball"]
                detections += detector.detect_ball_tiled(frame)

            # Update tracker (frame passed through for jersey-color team assignment)
            active_tracks = tracker.update(detections, frame_no, frame=frame)

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

        # Fold the re-ID gallery (players lost near the very end of the video, still within
        # their re-ID window) back into the active track set so their history isn't silently
        # dropped from the final player/stats output.
        tracker.tracks.update(tracker.lost_gallery)

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
