#!/usr/bin/env python3
"""
Re-runs the tutorial's detector + ByteTrack + pitch-keypoint homography (no team
classifier - that's the slow part) and caches per frame the pixel boxes, ball
candidates with confidence, and the averaged homography matrix. Lets later
post-processing (track-level team vote, ID-swap check crops, ball tracking)
work without redoing inference.

Tracker settings match roboflow_full_pipeline.py, so IDs should line up with
that run's frames.jsonl (checked by the post-processing script).

Usage: python roboflow_cache_detections.py <video> <out.jsonl>
"""
import sys
import json
from collections import deque

import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

VIDEO, OUT = sys.argv[1], sys.argv[2]
PLAYER_MODEL = r".local_data/models/football-player-detection.pt"
PITCH_MODEL = r".local_data/models/football-pitch-detection.pt"
BALL = 0
CONF = 0.3
KEYPOINT_CONF = 0.5
HOMOGRAPHY_WINDOW = 5

CONFIG = SoccerPitchConfiguration()


def main():
    det_model = YOLO(PLAYER_MODEL)
    pitch_model = YOLO(PITCH_MODEL)
    info = sv.VideoInfo.from_video_path(VIDEO)
    tracker = sv.ByteTrack(frame_rate=int(info.fps))
    pitch_vertices = np.array(CONFIG.vertices, dtype=np.float32)
    matrices = deque(maxlen=HOMOGRAPHY_WINDOW)
    last_m = None

    with open(OUT, "w") as f:
        for i, frame in enumerate(sv.get_video_frames_generator(VIDEO)):
            det = sv.Detections.from_ultralytics(det_model(frame, imgsz=1280, verbose=False)[0])
            det = det[det.confidence > CONF]
            ball = det[det.class_id == BALL]
            rest = det[det.class_id != BALL].with_nms(threshold=0.5, class_agnostic=True)
            rest = tracker.update_with_detections(rest)

            kp = sv.KeyPoints.from_ultralytics(pitch_model(frame, verbose=False)[0])
            if len(kp.xy):
                mask = kp.keypoint_confidence[0] > KEYPOINT_CONF
                if mask.sum() >= 4:
                    t = ViewTransformer(source=kp.xy[0][mask].astype(np.float32), target=pitch_vertices[mask])
                    matrices.append(t.m)
                    last_m = np.mean(np.array(matrices), axis=0)

            f.write(json.dumps({
                "frame": i,
                "tracks": [[int(t), int(c), *[round(float(v), 1) for v in xyxy], round(float(cf), 3)]
                           for t, c, xyxy, cf in zip(rest.tracker_id, rest.class_id, rest.xyxy, rest.confidence)],
                "ball": [[*[round(float(v), 1) for v in xyxy], round(float(cf), 3)]
                         for xyxy, cf in zip(ball.xyxy, ball.confidence)],
                "m": last_m.tolist() if last_m is not None else None,
            }) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"frame {i}/{info.total_frames}", flush=True)


if __name__ == "__main__":
    main()
