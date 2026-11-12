#!/usr/bin/env python3
"""
Recomputes PASS/SHOT counts for a tracking run using SigLIP-based team
clustering instead of the pipeline's own (fragile) HSV color clustering -
excludes tracks whose embedding doesn't clearly belong to either team
cluster (likely non-players: officials, staff, camera crew), then reuses the
REAL, already-fixed EventDetector to compute events, exactly like the
production pipeline does.

Usage: python count_passes_siglip.py <match_id> <pickle_path>
"""
import sys
import json
import pickle
import sqlite3
import bisect
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "backend")
sys.path.insert(0, r"C:\Users\Shayan Sethi\.cache\torch\hub\baudm_parseq_main")

import cv2
import numpy as np
import torch
from sports.common.team import TeamClassifier

from app.cv.tracker import TrackedObject
from app.cv.events import EventDetector
from app.cv.calibration import PitchCalibration

MATCH_ID = sys.argv[1]
PICKLE_PATH = sys.argv[2]
DB_PATH = r".local_data/gaffer_test.db"
VIDEO_PATH = r"test_videos\Barcelona vs Espanyol 2-1 1st half Tactical Cam - CameraTattica (1080p).mp4"
CROPS_PER_TRACK = 20
# How far (in the fitted KMeans's own distance units) an embedding can sit
# from its nearest team centroid before the track is excluded as not
# belonging to either real team - analogous to the old REFEREE_OUTLIER_DIST,
# but in SigLIP's much richer embedding space instead of raw hue/saturation.
OUTLIER_PERCENTILE = 85  # tracks beyond this percentile of nearest-centroid distance are excluded


def get_crops(cur, tid, n=CROPS_PER_TRACK):
    cur.execute(
        "SELECT frame_no, bbox FROM detections WHERE match_id=? AND track_id=? ORDER BY frame_no",
        (MATCH_ID, tid),
    )
    rows = cur.fetchall()
    dets = [(f, json.loads(b) if isinstance(b, str) else b) for f, b in rows]
    if not dets:
        return []
    idxs = sorted(set(int(round(i * (len(dets) - 1) / (n - 1))) for i in range(min(n, len(dets)))))
    crops = []
    cap = get_crops.cap
    for idx in idxs:
        frame_no, bbox = dets[idx]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ret, frame = cap.read()
        if not ret:
            continue
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        h = crop.shape[0]
        torso = crop[int(h * 0.1):int(h * 0.65), :]
        if torso.size > 0:
            crops.append(torso)
    return crops


def main():
    with open(PICKLE_PATH, "rb") as f:
        payload = pickle.load(f)
    raw_tracks = payload["tracks"]
    player_track_ids = [int(tid) for tid, t in raw_tracks.items() if t["class_name"] == "player"]
    print(f"total player tracks: {len(player_track_ids)}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    get_crops.cap = cv2.VideoCapture(VIDEO_PATH)

    # --- 1. Pull crops and fit the SigLIP team classifier across ALL tracks ---
    track_crops: Dict[int, List[np.ndarray]] = {}
    all_crops = []
    crop_owner = []
    for tid in player_track_ids:
        crops = get_crops(cur, tid)
        track_crops[tid] = crops
        all_crops.extend(crops)
        crop_owner.extend([tid] * len(crops))
        print(f"  track {tid}: {len(crops)} crops", flush=True)

    print(f"\nFitting SigLIP TeamClassifier on {len(all_crops)} crops...", flush=True)
    clf = TeamClassifier(device="cpu")
    clf.fit(all_crops)

    # Re-derive per-crop distance to nearest cluster centroid via the
    # classifier's own fitted UMAP+KMeans, so outlier tracks (officials,
    # staff) can be excluded the same principled way REFEREE_OUTLIER_DIST
    # did for the old HSV approach - just in a far richer feature space.
    features = clf.extract_features(all_crops)
    reduced = clf.reducer.transform(features)
    cluster_labels = clf.cluster_model.predict(reduced)
    centroids = clf.cluster_model.cluster_centers_
    dists = np.linalg.norm(reduced - centroids[cluster_labels], axis=1)

    track_cluster: Dict[int, int] = {}
    track_dist: Dict[int, float] = {}
    idx = 0
    for tid in player_track_ids:
        n = len(track_crops[tid])
        if n == 0:
            track_cluster[tid] = -1
            track_dist[tid] = 1e9
            continue
        tid_labels = cluster_labels[idx:idx + n]
        tid_dists = dists[idx:idx + n]
        idx += n
        vals, counts = np.unique(tid_labels, return_counts=True)
        track_cluster[tid] = int(vals[np.argmax(counts)])
        track_dist[tid] = float(np.median(tid_dists))

    valid_dists = [d for tid, d in track_dist.items() if d < 1e9]
    outlier_cutoff = np.percentile(valid_dists, OUTLIER_PERCENTILE)
    print(f"\noutlier cutoff (p{OUTLIER_PERCENTILE}): {outlier_cutoff:.3f}")

    excluded = {tid for tid, d in track_dist.items() if d > outlier_cutoff}
    included = [tid for tid in player_track_ids if tid not in excluded]
    print(f"excluded as likely non-players ({len(excluded)}): {sorted(excluded)}")
    print(f"included as real players ({len(included)}): {sorted(included)}")

    # cluster 0 / 1 -> HOME / AWAY (arbitrary which is which at this point)
    for tid in included:
        raw_tracks[str(tid)]["team"] = "HOME" if track_cluster[tid] == 0 else "AWAY"
    for tid in excluded:
        raw_tracks[str(tid)]["team"] = "REFEREE"

    with open(r"C:\Users\SHAYAN~1\AppData\Local\Temp\claude\C--Users-Shayan-Sethi-gaffer\83fe6ffb-4aa4-48f0-bbd5-333435412950\scratchpad\siglip_team_assignment.json", "w") as f:
        json.dump({
            "excluded": sorted(excluded),
            "included_home": sorted([t for t in included if track_cluster[t] == 0]),
            "included_away": sorted([t for t in included if track_cluster[t] == 1]),
            "track_dist": {str(k): v for k, v in track_dist.items()},
        }, f, indent=2)

    # --- 2. Run the REAL EventDetector with corrected team labels ---
    total_frames = payload.get("total_frames", 150)
    fps = payload.get("fps", 30.0)
    width = payload.get("frame_width", 1280)
    height = payload.get("frame_height", 720)
    pitch_bounds = payload.get("pitch_bounds")
    pitch_bounds_history = payload.get("pitch_bounds_history") or []
    pbh_frames = [f for f, _ in pitch_bounds_history]
    home_attacks_right = payload.get("home_attacks_right", True)
    goal_axis = payload.get("goal_axis", "x")
    pitch_length_m = payload.get("pitch_length_m", 40.0)
    pitch_width_m = payload.get("pitch_width_m", 20.0)
    calibration = PitchCalibration.from_points_json(
        payload.get("calibration_points"), pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m
    )

    tracked_objects: Dict[int, TrackedObject] = {}
    track_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}
    track_frames_sorted: Dict[int, List[int]] = {}
    for t_id_str, t_data in raw_tracks.items():
        t_id = int(t_id_str)
        if t_data["class_name"] == "player" and t_data["team"] == "REFEREE":
            continue  # excluded non-player - never reaches event detection
        obj = TrackedObject(
            track_id=t_id, class_name=t_data["class_name"], bbox=[0, 0, 0, 0],
            centroid=tuple(t_data["history"][-1][1]) if t_data["history"] else (0, 0), frame_no=0,
        )
        obj.team = t_data["team"]
        obj.history = [(f, tuple(pt)) for f, pt in t_data["history"]]
        tracked_objects[t_id] = obj
        positions_by_frame = {f: pt for f, pt in obj.history}
        track_positions[t_id] = positions_by_frame
        track_frames_sorted[t_id] = sorted(positions_by_frame.keys())

    max_gap_ball = int(fps * 3)
    max_gap_player = int(fps * 1)

    def position_at_frame(frames_sorted, positions, frame_no, max_gap):
        if frame_no in positions:
            return positions[frame_no]
        i = bisect.bisect_left(frames_sorted, frame_no)
        candidates = []
        if i > 0:
            candidates.append(frames_sorted[i - 1])
        if i < len(frames_sorted):
            candidates.append(frames_sorted[i])
        if not candidates:
            return None
        nearest = min(candidates, key=lambda f: abs(f - frame_no))
        if abs(nearest - frame_no) > max_gap:
            return None
        return positions[nearest]

    event_detector = EventDetector(
        frame_width=width, frame_height=height, fps=fps,
        pitch_bounds=tuple(pitch_bounds) if pitch_bounds else None,
        pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
        calibration=calibration, home_attacks_right=home_attacks_right, goal_axis=goal_axis,
    )

    all_events = []
    for frame_no in range(1, total_frames + 1):
        frame_tracks = []
        for t_id, obj in tracked_objects.items():
            max_gap = max_gap_ball if obj.class_name == "ball" else max_gap_player
            pt = position_at_frame(track_frames_sorted[t_id], track_positions[t_id], frame_no, max_gap)
            if pt is None:
                continue
            current_t = TrackedObject(
                track_id=obj.track_id, class_name=obj.class_name,
                bbox=[pt[0] - 20, pt[1] - 40, pt[0] + 20, pt[1]], centroid=pt, frame_no=frame_no,
            )
            current_t.team = obj.team
            frame_tracks.append(current_t)

        if pbh_frames:
            idx2 = min(bisect.bisect_right(pbh_frames, frame_no), len(pbh_frames)) - 1
            if idx2 >= 0:
                event_detector.update_pitch_bounds(tuple(pitch_bounds_history[idx2][1]))

        frame_evts = event_detector.process_frame(frame_tracks, frame_no)
        all_events.extend(frame_evts)

    print(f"\n{len(all_events)} total events")
    from collections import Counter
    by_type_team = Counter((e.event_type, e.team) for e in all_events)
    for (etype, team), count in sorted(by_type_team.items()):
        print(f"  {etype:20} {team:8} {count}")

    print("\nSHOT events:")
    for e in all_events:
        if e.event_type == "SHOT":
            print(f"  [{e.frame_no/fps:.1f}s] {e.team} {e.data_json}")

    print("\nPASS events (first 30):")
    n_printed = 0
    for e in all_events:
        if e.event_type == "PASS":
            print(f"  [{e.frame_no/fps:.1f}s] {e.team} {e.data_json}")
            n_printed += 1
            if n_printed >= 30:
                break


if __name__ == "__main__":
    main()
