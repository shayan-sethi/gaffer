#!/usr/bin/env python3
"""
Scores the tutorial's detector + ByteTrack on a SoccerTrack v2 clip against its
22-player MOT ground truth, plain vs tiled full-resolution detection.

  plain : whole frame resized to 1280 (what roboflow_full_pipeline.py does)
  tiled : 1280x1280 crops at native resolution, merged with class-agnostic NMS

Reports recall (IoU >= 0.3, by GT box height) and ID switches / fragmentation
for the 22 labelled players. Precision is not reported: the GT labels only the
22 players, so coaches/bench/referees would count as false positives.

Usage: python soccertrack_benchmark.py <clip.mp4> <gt.txt> <n_frames> <plain|tiled> <out.json>
"""
import sys
import json
import time

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

CLIP, GT, N, MODE, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
MODEL = r".local_data/models/football-player-detection.pt"
CONF = 0.25
IOU_MATCH = 0.3
TILE, TILE_STEP = 1280, 1000
BAND_Y0 = 480  # tiles cover y in [BAND_Y0, BAND_Y0 + TILE): the pitch band of this camera


def detect(model, frame):
    if MODE == "plain":
        return sv.Detections.from_ultralytics(model(frame, imgsz=1280, verbose=False, conf=CONF)[0])
    h, w = frame.shape[:2]
    y0 = min(BAND_Y0, max(0, h - TILE))
    parts = []
    xs = list(range(0, max(1, w - TILE + 1), TILE_STEP))
    if xs[-1] != w - TILE:
        xs.append(w - TILE)
    for x0 in xs:
        crop = frame[y0:y0 + TILE, x0:x0 + TILE]
        d = sv.Detections.from_ultralytics(model(crop, imgsz=1280, verbose=False, conf=CONF)[0])
        d.xyxy = d.xyxy + np.array([x0, y0, x0, y0], dtype=np.float32)
        parts.append(d)
    return sv.Detections.merge(parts).with_nms(threshold=0.5, class_agnostic=True)


def iou_matrix(a, b):
    ax1, ay1, ax2, ay2 = a[:, None, 0], a[:, None, 1], a[:, None, 2], a[:, None, 3]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / np.maximum(ua, 1e-9)


def main():
    gt = np.loadtxt(GT, delimiter=",")
    model = YOLO(MODEL)
    tracker = sv.ByteTrack(frame_rate=25)
    cap = cv2.VideoCapture(CLIP)
    matched_ids = {}  # gt id -> list of (frame, tracker id)
    hit = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    t0 = time.time()
    n_det = 0
    for f in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        det = detect(model, frame)
        n_det += len(det)
        tracked = tracker.update_with_detections(det[det.class_id != 0])
        g = gt[gt[:, 0] == f]
        gxyxy = np.stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]], axis=1)
        if len(tracked):
            iou = iou_matrix(gxyxy, tracked.xyxy)
            r, c = linear_sum_assignment(-iou)
            pairs = {int(a): int(b) for a, b in zip(r, c) if iou[a, b] >= IOU_MATCH}
        else:
            pairs = {}
        for k in range(len(g)):
            bucket = "small" if g[k, 5] < 25 else "medium" if g[k, 5] < 40 else "large"
            hit[bucket][1] += 1
            if k in pairs:
                hit[bucket][0] += 1
                matched_ids.setdefault(int(g[k, 1]), []).append((f, int(tracked.tracker_id[pairs[k]])))
        if f % 25 == 0:
            print(f"frame {f}/{N}  {time.time() - t0:.0f}s", flush=True)

    switches = frag = 0
    for gid, seq in matched_ids.items():
        ids = [t for _, t in seq]
        switches += sum(1 for a, b in zip(ids, ids[1:]) if a != b)
        frames = [fr for fr, _ in seq]
        frag += sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
    tot = sum(v[1] for v in hit.values())
    res = {
        "mode": MODE, "frames": N, "seconds": round(time.time() - t0),
        "sec_per_frame": round((time.time() - t0) / max(1, N), 2),
        "recall_all": round(sum(v[0] for v in hit.values()) / max(1, tot), 4),
        "recall_by_height": {k: round(v[0] / max(1, v[1]), 4) for k, v in hit.items()},
        "gt_boxes_by_height": {k: v[1] for k, v in hit.items()},
        "id_switches": switches, "track_fragments": frag,
        "distinct_tracker_ids_on_gt": len({t for s in matched_ids.values() for _, t in s}),
        "gt_players_ever_matched": len(matched_ids),
        "detections_per_frame": round(n_det / max(1, N), 1),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
