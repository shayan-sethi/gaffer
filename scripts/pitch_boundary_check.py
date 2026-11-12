#!/usr/bin/env python3
"""
Uses the pretrained football-pitch-detection keypoint model to compute the
REAL pitch geometry automatically (no manual corner-picking), then checks
every player detection against it - excluding tracks that spend most of
their life outside the actual pitch boundary (bench staff, camera crew),
which color/embedding-based approaches couldn't reliably catch.
"""
import sys
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import sports.configs.soccer as pitch_cfg

MATCH_ID = sys.argv[1]
DB_PATH = r".local_data/gaffer_test.db"
VIDEO_PATH = r"test_videos\Barcelona vs Espanyol 2-1 1st half Tactical Cam - CameraTattica (1080p).mp4"
PITCH_MODEL_PATH = r".local_data/models/football-pitch-detection.pt"
CALIB_SAMPLE_INTERVAL_FRAMES = 25  # ~once per second at 25fps
KEYPOINT_CONF = 0.5
MAX_FRAMES = 4500  # only the window we actually tracked - was accidentally the whole video
# How far past the real pitch rectangle (in real-world cm) a detection is
# still considered "on the pitch". Measured directly on real footage: 4
# confirmed real players NEVER registered outside the rectangle at all
# (exactly 0cm on every sample, even near the touchline) - unlike a margin
# guessed in advance, this is closer to the true value: real players simply
# don't leave the pitch, so almost any nonzero margin already separates them
# from confirmed non-players (linesmen, staff), who register outside by
# 15-330cm on a large fraction of their samples.
MARGIN_CM = 50
# What fraction of a track's samples are allowed to sit outside the margin
# before it's flagged as not a genuine on-pitch player - real players
# measured 0% (never), confirmed non-players measured 40-70%.
MAX_OUTSIDE_FRACTION = 0.05


def main():
    model = YOLO(PITCH_MODEL_PATH)
    cfg = pitch_cfg.SoccerPitchConfiguration()
    vertices = np.array(cfg.vertices, dtype=np.float32)  # (32, 2) in cm

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- 1. Sample calibrations across the video ---
    calibrations: List[Tuple[int, Optional[np.ndarray]]] = []  # (frame_no, homography or None)
    frame_no = 0
    while frame_no < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break
        result = model(frame, verbose=False)[0]
        kp = result.keypoints
        H = None
        if kp is not None and kp.conf is not None:
            xy = kp.xy[0].numpy()
            conf = kp.conf[0].numpy()
            mask = conf > KEYPOINT_CONF
            if mask.sum() >= 4:
                src = xy[mask].astype(np.float32)
                dst = vertices[mask]
                H, _ = cv2.findHomography(src, dst)
        calibrations.append((frame_no, H))
        frame_no += CALIB_SAMPLE_INTERVAL_FRAMES
        if frame_no % 250 < CALIB_SAMPLE_INTERVAL_FRAMES:
            print(f"  calibrated up to frame {frame_no}/{total_frames}", flush=True)

    n_ok = sum(1 for _, H in calibrations if H is not None)
    print(f"\n{n_ok}/{len(calibrations)} calibration snapshots succeeded (>=4 confident keypoints)")

    calib_frames = [f for f, H in calibrations]
    calib_H = [H for f, H in calibrations]

    def nearest_homography(frame_no):
        idx = min(range(len(calib_frames)), key=lambda i: abs(calib_frames[i] - frame_no))
        return calib_H[idx]

    # --- 2. Check every player detection against the pitch boundary ---
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT track_id FROM detections WHERE match_id=? AND class_name='player' ORDER BY track_id",
        (MATCH_ID,),
    )
    track_ids = [r[0] for r in cur.fetchall()]

    results = {}
    for tid in track_ids:
        cur.execute(
            "SELECT frame_no, bbox FROM detections WHERE match_id=? AND track_id=? AND frame_no<=? ORDER BY frame_no",
            (MATCH_ID, tid, MAX_FRAMES),
        )
        rows = cur.fetchall()
        in_bounds = 0
        out_bounds = 0
        no_calib = 0
        for frame_no, bbox_json in rows:
            bbox = json.loads(bbox_json) if isinstance(bbox_json, str) else bbox_json
            foot_x, foot_y = (bbox[0] + bbox[2]) / 2.0, bbox[3]
            H = nearest_homography(frame_no)
            if H is None:
                no_calib += 1
                continue
            pt = cv2.perspectiveTransform(np.array([[[foot_x, foot_y]]], dtype=np.float32), H)
            wx, wy = pt[0, 0]
            if -MARGIN_CM <= wx <= cfg.length + MARGIN_CM and -MARGIN_CM <= wy <= cfg.width + MARGIN_CM:
                in_bounds += 1
            else:
                out_bounds += 1
        total = in_bounds + out_bounds
        frac_out = out_bounds / total if total else 1.0
        is_player = frac_out <= MAX_OUTSIDE_FRACTION
        results[tid] = {
            "in_bounds": in_bounds, "out_bounds": out_bounds, "no_calib": no_calib,
            "frac_out": frac_out, "is_player": is_player,
        }

    print(f"\n{'track':6} {'in':6} {'out':6} {'no_calib':9} {'frac_out':9} {'verdict'}")
    for tid in sorted(results, key=lambda t: -results[t]["frac_out"]):
        r = results[tid]
        verdict = "PLAYER" if r["is_player"] else "EXCLUDE"
        print(f"{tid:6} {r['in_bounds']:6} {r['out_bounds']:6} {r['no_calib']:9} {r['frac_out']:.3f}     {verdict}")

    n_excluded = sum(1 for r in results.values() if not r["is_player"])
    print(f"\n{n_excluded} excluded, {len(results) - n_excluded} kept as real players")

    out_path = r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\pitch_boundary_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
