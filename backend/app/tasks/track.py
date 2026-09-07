import os
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import Match, MatchStatus, Player, Detection
from app.cv.calibration import PitchCalibration
from app.cv.detector import YOLOv8Detector
from app.cv.tracker import CentroidTracker
from app.cv.bot_sort_tracker import BotSortPlayerTracker
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

        # Carried through to the events/stats tasks so they can build a
        # homography-corrected PitchCalibration instead of the linear
        # pixel-to-meter approximation, if this fixed camera has been
        # calibrated (see app.cv.calibration.PitchCalibration).
        calibration_points = match.camera.calibration_points if match.camera else None

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
        # reid_gallery_max_age is in real video frame units (frame_no keeps counting
        # every frame even though we only process every `sampled_rate`-th one). At a
        # typical 24-30fps this needs to be large - a match-length window, not a
        # few-second one - since a player can be occluded, off-camera, or briefly
        # subbed for minutes and should still be recognized as the same person when
        # they reappear rather than minted as a new track. 18000 frames is ~12.5
        # minutes at 24fps.
        tracker = CentroidTracker(max_lost_frames=45, max_distance_px=90.0, reid_gallery_max_age=18000)

        # Opt-in alternative to the hand-rolled CentroidTracker above, using
        # ultralytics' built-in BoT-SORT tracker for PLAYERS only (see
        # BotSortPlayerTracker's docstring for why not the ball too). Default
        # stays "centroid" - this is still an experimental path being evaluated
        # against known real footage, not yet the production default.
        tracker_backend = os.getenv("TRACKER_BACKEND", "centroid")
        bot_sort_tracker: Optional[BotSortPlayerTracker] = None
        ball_tracker: Optional[CentroidTracker] = None
        if tracker_backend == "botsort":
            bot_sort_tracker = BotSortPlayerTracker(
                model_path=settings.YOLO_MODEL_PATH,
                class_map=detector.class_map,
                conf_threshold=settings.CONFIDENCE_THRESHOLD,
                iou_threshold=settings.IOU_THRESHOLD,
                img_size=settings.YOLO_IMG_SIZE,
                device=detector.device,
            )
            # The ball still gets CentroidTracker's existing single-ball continuity
            # logic (BoT-SORT has no concept of "only one real ball" - see
            # BotSortPlayerTracker docstring), just fed ball-only detections so its
            # player-matching code path never runs. Track IDs offset well clear of
            # BoT-SORT's own player IDs (which start at 1 and stay in the low
            # thousands for any real match) so merging the two trackers' `.tracks`
            # dicts at the end can never silently collide/overwrite.
            ball_tracker = CentroidTracker(ball_max_lost_frames=300)
            ball_tracker.next_track_id = 1_000_000

        frame_no = 0
        all_detections_batch: List[Detection] = []
        sampled_rate = 2  # Process every 2nd frame for speed/efficiency while maintaining tracks
        # Re-detected every processed frame, not computed once and locked in - the
        # camera pans/zooms to follow the ball rather than staying static, so a
        # mask locked to wherever the pitch sat in an early frame would silently
        # start letting sideline spectators/bystanders count as "players" (or
        # clipping real ones) the moment the camera moves. Cheap enough (~8ms vs
        # ~50ms of YOLO inference per frame) that recomputing it every frame costs
        # far less than the correctness it buys.
        pitch_mask: PitchMask = None
        last_valid_bbox: tuple = None  # falls back to this on a frame segmentation fails for
        # Per-processed-frame samples of the pitch region above, so downstream
        # goal/corner zone checks can track the camera's pan over the match
        # instead of freezing on whichever single frame's segmentation happened
        # to be the last one recorded (see EventDetector.update_pitch_bounds -
        # a static single snapshot let a panning camera silently invalidate goal
        # detection for the rest of the match).
        pitch_bounds_history: List[Tuple[int, List[float]]] = []

        max_limit = int(os.getenv("MAX_FRAMES", "0"))
        # Absolute 1-indexed frame number to start processing from (same convention
        # as MAX_FRAMES, which is an absolute end frame, not a count) - lets a caller
        # isolate an arbitrary window (e.g. minutes 5-10) without paying for detection/
        # tracking over the frames before it. Seeking past them is near-free; running
        # YOLO over them just to throw the results away is not.
        start_frame = int(os.getenv("START_FRAME", "0"))
        # Which team attacks which goal - purely a camera/kickoff convention for
        # this recording (see EventDetector.home_attacks_right), not something
        # detectable from the footage itself, so it's operator-supplied the same
        # way START_FRAME/MAX_FRAMES are for this manual validation workflow.
        home_attacks_right = os.getenv("HOME_ATTACKS_RIGHT", "1") != "0"
        # Which pixel axis the goals sit along for this camera - "x" (default,
        # sideline camera, goals at left/right) or "y" (mounted behind/above
        # one goal, shooting down the pitch length - see EventDetector.goal_axis).
        goal_axis = os.getenv("GOAL_AXIS", "x")
        # Real pitch dimensions - default to the 5-a-side turf size this
        # pipeline was built against; a full-size 11-a-side pitch needs the
        # real ~105x68m or every distance/speed/xG figure and the corner/
        # throw-in radius checks scale against the wrong pitch entirely.
        pitch_length_m = float(os.getenv("PITCH_LENGTH_M", "40.0"))
        pitch_width_m = float(os.getenv("PITCH_WIDTH_M", "20.0"))
        # A second, geometric "is this actually on the pitch" check, independent
        # of PitchMask's color-based one - only available once the camera's
        # been calibrated (see PitchCalibration.contains_point_px's docstring
        # for why the color mask alone isn't enough: it can't reliably tell a
        # technical-area strip from the pitch itself when both read as
        # grass-colored). None when uncalibrated, in which case the color mask
        # below is the only filter, same as before this existed.
        pitch_calibration = PitchCalibration.from_points_json(calibration_points, pitch_length_m, pitch_width_m)
        # How far past the real pitch rectangle a detection is still allowed
        # to land - real throw-ins/corners happen right at the line, and
        # calibration from 4 hand-picked corners isn't pixel-perfect, but a
        # technical area sits meters further back than that.
        PITCH_CALIBRATION_MARGIN_M = 3.0

        def _within_calibrated_pitch(cx: float, cy: float) -> bool:
            if pitch_calibration is None:
                return True
            return pitch_calibration.contains_point_px(cx, cy, margin_m=PITCH_CALIBRATION_MARGIN_M)

        if start_frame > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
            frame_no = start_frame - 1
        effective_total = min(total_frames, max_limit) if max_limit > 0 else total_frames
        logger.info(
            f"Processing frames {max(1, start_frame)}-{effective_total} for match_id={match_id} "
            f"(fps={fps}, res={width}x{height}, GPU={detector.device})..."
        )

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
                player_track_count = (
                    len(bot_sort_tracker.tracks) if bot_sort_tracker is not None
                    else len([t for t in tracker.tracks.values() if t.class_name == "player"])
                )
                logger.info(f"CV Tracking Progress: Frame {frame_no}/{effective_total} ({pct:.1f}%) | Active Player Tracks: {player_track_count}")

            timestamp_ms = int((frame_no / fps) * 1000)

            pitch_mask = PitchMask(frame)
            pitch_bbox = pitch_mask.bounding_box()
            if pitch_bbox is None and last_valid_bbox is not None:
                # A single frame's segmentation can fail transiently (fast zoom,
                # motion blur, a glare) even mid-pan when the pitch is genuinely
                # still in frame - reuse the last confident detection for this one
                # frame rather than disabling ROI filtering entirely and letting
                # every bystander through.
                pitch_bbox = last_valid_bbox
            elif pitch_bbox is not None:
                last_valid_bbox = pitch_bbox

            if pitch_bbox is not None:
                pitch_bounds_history.append((frame_no, list(pitch_bbox)))
                px1, py1, px2, py2 = pitch_bbox
                # Generous padding (not the tight grass contour) - a real ball
                # legitimately leaves the grass mask at goal-mouths, corners, and
                # throw-ins. This is only meant to reject detections clearly off
                # in the crowd/stands, not to clip real goal/boundary play.
                pad_x, pad_y = (px2 - px1) * 0.15, (py2 - py1) * 0.25
                ball_bounds = (px1 - pad_x, py1 - pad_y, px2 + pad_x, py2 + pad_y)
            else:
                ball_bounds = None

            if bot_sort_tracker is not None:
                # BoT-SORT path: player detection + tracking happen together in
                # one model.track() call, including pitch_mask filtering (see
                # BotSortPlayerTracker.update). The ball isn't handed to BoT-SORT
                # (see its docstring for why) - its raw per-frame boxes still go
                # through the same ball_bounds filtering, tiled-fallback, and
                # CentroidTracker single-ball continuity as the path below.
                player_active, player_dets, ball_dets = bot_sort_tracker.update(
                    frame, frame_no, pitch_mask=pitch_mask, within_pitch=_within_calibrated_pitch
                )

                if ball_bounds is not None:
                    bx1, by1, bx2, by2 = ball_bounds
                    ball_dets = [
                        d for d in ball_dets
                        if bx1 <= (d.bbox[0] + d.bbox[2]) / 2.0 <= bx2
                        and by1 <= (d.bbox[1] + d.bbox[3]) / 2.0 <= by2
                    ]

                main_pass_found_ball = len(ball_dets) > 0
                if not is_synthetic_source and not main_pass_found_ball:
                    ball_dets = ball_dets + detector.detect_ball_tiled(frame)

                ball_active = ball_tracker.update(ball_dets, frame_no)
                active_tracks = player_active + ball_active
                detections = player_dets + ball_dets
            else:
                # Detect objects (players and ball)
                detections = detector.detect_frame(frame)

                # Drop "player" detections that fall outside the field of play - sideline
                # spectators, benches, and bystanders are detected by YOLO just as readily
                # as players, and otherwise fragment/inflate track IDs.
                detections = [
                    d for d in detections
                    if d.class_name != "player" or (
                        pitch_mask.contains_point((d.bbox[0] + d.bbox[2]) / 2.0, d.bbox[3])
                        and _within_calibrated_pitch((d.bbox[0] + d.bbox[2]) / 2.0, d.bbox[3])
                    )
                ]

                # Same idea for the ball, but padded generously (see above) - a tiled
                # low-confidence false positive up in the crowd/stands is exactly what
                # was hijacking the ball track and firing fake goal events.
                if ball_bounds is not None:
                    bx1, by1, bx2, by2 = ball_bounds
                    detections = [
                        d for d in detections
                        if d.class_name != "ball" or (
                            bx1 <= (d.bbox[0] + d.bbox[2]) / 2.0 <= bx2
                            and by1 <= (d.bbox[1] + d.bbox[3]) / 2.0 <= by2
                        )
                    ]

                # A single full-frame pass essentially never finds the ball on real broadcast-
                # distance footage - it's too small/low-contrast. Tiled ball-only detection
                # trades extra inference passes (4x by default) for enough effective
                # resolution to see it - but that's only needed when the cheap full-frame
                # pass actually missed it. Skip this on the synthetic demo canvas: the
                # fallback already places a simulated ball there, and there's no real ball
                # to tile-search for.
                main_pass_found_ball = any(d.class_name == "ball" for d in detections)
                if not is_synthetic_source and not main_pass_found_ball:
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

        if bot_sort_tracker is not None:
            # Drop single-frame ghost tracks (a false-positive detection BoT-SORT
            # still confidently minted an ID for) before merging - see
            # prune_unconfirmed_tracks's docstring for why this has to happen once,
            # at the very end, rather than continuously during the loop above.
            bot_sort_tracker.prune_unconfirmed_tracks()
            # Two independent trackers this run (players via BoT-SORT, ball via
            # CentroidTracker) - merge into one dict for everything downstream.
            # Track-ID namespaces can't collide (see ball_tracker.next_track_id
            # above), so a plain dict merge is safe.
            all_tracks = {**bot_sort_tracker.tracks, **ball_tracker.tracks}
        else:
            # Fold the re-ID gallery (players lost near the very end of the video, still within
            # their re-ID window) back into the active track set so their history isn't silently
            # dropped from the final player/stats output.
            tracker.tracks.update(tracker.lost_gallery)
            all_tracks = tracker.tracks

        # Batch insert detections
        if all_detections_batch:
            db.bulk_save_objects(all_detections_batch)
            db.commit()

        # Create Player records for all player tracks found - excluding any track
        # _assign_teams identified as a referee (color sits far from BOTH team
        # clusters; see TeamAssignmentMixin.REFEREE_OUTLIER_DIST), so a referee
        # never gets a Player record, a jersey number, or counts toward any
        # team's passes/shots/possession/distance stats.
        player_tracks = {
            t_id: t for t_id, t in all_tracks.items()
            if t.class_name == "player" and t.team != "REFEREE"
        }
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

        # Prepare serializable summary for downstream tasks - referee tracks are
        # dropped here too (not just from Player records above), so they never
        # reach event/stats reconstruction as a "player" at all: EventDetector's
        # nearest-player-to-ball logic has no notion of "ignore this one", so a
        # referee left in would still count for possession/pass/shot purposes.
        serializable_tracks = {}
        for t_id, t in all_tracks.items():
            if t.class_name == "player" and t.team == "REFEREE":
                continue
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
            # Last frame's pitch region - a static snapshot even though the mask
            # itself was tracked live throughout (see the per-frame recompute
            # above). Downstream event/stats zone math (goals, corners, thirds,
            # linear speed scaling) still assumes one fixed pitch position for
            # the whole match; that's a separate, larger gap a panning camera
            # exposes, not fixed by live-tracking the ROI mask alone.
            "pitch_bounds": list(last_valid_bbox) if last_valid_bbox is not None else None,
            "pitch_bounds_history": pitch_bounds_history,
            "calibration_points": calibration_points,
            "home_attacks_right": home_attacks_right,
            "goal_axis": goal_axis,
            "pitch_length_m": pitch_length_m,
            "pitch_width_m": pitch_width_m,
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
