#!/usr/bin/env python3
"""
Team-agnostic per-individual pass counting: tags every tracked player by
their own shirt color (no HOME/AWAY clustering) and counts ball-possession
transfers between individuals directly, regardless of team.
"""
import sys
import pickle
import sqlite3
import json
import cv2
import numpy as np

PICKLE_PATH = sys.argv[1]
VIDEO_PATH = sys.argv[2]
OUT_GRID = sys.argv[3]
DB_PATH = sys.argv[4]
MATCH_ID = sys.argv[5]
POSSESSION_RADIUS_PX = 65.0


HUE_NAMES = [
    (10, "red"), (22, "orange"), (34, "yellow"), (85, "green"),
    (100, "cyan"), (135, "blue"), (155, "purple"), (170, "pink"), (180, "red"),
]


def _hue_to_name(h):
    for upper, name in HUE_NAMES:
        if h < upper:
            return name
    return "red"


def crop_color_name(crop):
    """
    Classifies a single crop's dominant shirt color via a saturation-gated
    hue MODE, not a plain mean. A plain mean over hue+saturation is not
    robust here: an achromatic (white/grey/black) shirt has effectively
    random, meaningless hue at low saturation, while a MINORITY of
    high-saturation background pixels (grass bleeding into the crop, or a
    loose detection box) have a strong, consistent hue - so a simple
    circular/linear average lets that coherent minority dominate a
    scattered majority. Confirmed on real footage: every player in a game
    with visibly white/black/grey/neon shirts came back "green" under a
    plain mean. Fix: first check whether most pixels are even chromatic at
    all (saturation gate); only take a hue reading from the pixels that
    have one.
    """
    hsv = crop.reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    valid = v > 55  # dark/shadow pixels carry no reliable color info
    if valid.sum() < 10:
        return None
    h, s, v = h[valid], s[valid], v[valid]

    chromatic = s > 55
    achromatic_frac = 1.0 - chromatic.mean()
    if achromatic_frac > 0.55:
        mean_v = v[~chromatic].mean() if (~chromatic).any() else v.mean()
        if mean_v < 90:
            return "black"
        if mean_v < 185:
            return "grey"
        return "white"

    # Hue mode (histogram peak) among chromatic pixels only - robust to a
    # scattered achromatic minority the way a plain mean would not be.
    hist, edges = np.histogram(h[chromatic], bins=18, range=(0, 180))
    peak_hue = edges[np.argmax(hist)] + 5
    return _hue_to_name(peak_hue)


def sample_track_color(cap, detections, num_samples=8):
    """detections: list of (frame_no, [x1,y1,x2,y2]) - the REAL per-frame YOLO
    box, not a guessed fixed-size crop, so the torso region actually scales
    with how big the player is in frame (close to camera vs far away).
    Returns the majority-vote color name across several samples, plus the
    per-sample votes for transparency."""
    n = len(detections)
    if n == 0:
        return None, []
    idxs = sorted(set(int(round(i * (n - 1) / (num_samples - 1))) for i in range(min(num_samples, n))))
    votes = []
    for idx in idxs:
        frame_no, bbox = detections[idx]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ret, frame = cap.read()
        if not ret:
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        h = y2 - y1
        if h <= 0 or x2 <= x1:
            continue
        # torso band: skip head (~top 15%) and legs (below ~55%), same
        # proportions as the production _sample_color, but against the REAL
        # detected box instead of a guessed fixed-size crop.
        ty1, ty2 = y1 + int(h * 0.15), y1 + int(h * 0.55)
        torso = frame[ty1:ty2, x1:x2]
        if torso.size == 0:
            continue
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        name = crop_color_name(hsv)
        if name:
            votes.append(name)
    if not votes:
        return None, []
    from collections import Counter
    winner = Counter(votes).most_common(1)[0][0]
    return winner, votes


def main():
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)
    tracks = data["tracks"]
    player_tracks = {int(tid): t for tid, t in tracks.items() if t["class_name"] == "player"}
    ball_track = next(t for t in tracks.values() if t["class_name"] == "ball")
    ball_hist = ball_track["history"]

    cap = cv2.VideoCapture(VIDEO_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    detections_by_track = {}
    for tid in player_tracks:
        cur.execute(
            "SELECT frame_no, bbox FROM detections WHERE match_id=? AND track_id=? ORDER BY frame_no",
            (MATCH_ID, tid),
        )
        rows = cur.fetchall()
        detections_by_track[tid] = [(f, json.loads(b) if isinstance(b, str) else b) for f, b in rows]

    # --- tag every player by their own shirt color, no team clustering ---
    tags = {}
    vote_log = {}
    for tid, t in player_tracks.items():
        name, votes = sample_track_color(cap, detections_by_track.get(tid, []))
        tags[tid] = name or "unknown"
        vote_log[tid] = votes

    # disambiguate duplicate color names by order of first appearance
    first_seen = {tid: t["history"][0][0] for tid, t in player_tracks.items()}
    by_name = {}
    for tid in sorted(tags, key=lambda x: first_seen[x]):
        by_name.setdefault(tags[tid], []).append(tid)
    display_tag = {}
    for name, tids in by_name.items():
        if len(tids) == 1:
            display_tag[tids[0]] = name.capitalize()
        else:
            for i, tid in enumerate(tids, start=1):
                display_tag[tid] = f"{name.capitalize()}-{i}"

    # --- possession-transfer detection: nearest player to ball each sample,
    # log a transfer whenever that identity changes, regardless of team ---
    def nearest_player_at(frame_no, bx, by):
        best_tid, best_dist = None, POSSESSION_RADIUS_PX
        for tid, t in player_tracks.items():
            # nearest history sample to this frame_no
            hist = t["history"]
            # binary-search-ish: history is sorted by frame_no
            closest = min(hist, key=lambda h: abs(h[0] - frame_no))
            if abs(closest[0] - frame_no) > 15:  # ~0.5s at 30fps - too stale
                continue
            px, py = closest[1]
            d = np.hypot(px - bx, py - by)
            if d < best_dist:
                best_dist, best_tid = d, tid
        return best_tid

    passes_made = {tid: 0 for tid in player_tracks}
    passes_received = {tid: 0 for tid in player_tracks}
    transfer_log = []
    last_possessor = None
    for frame_no, (bx, by) in ball_hist:
        cur = nearest_player_at(frame_no, bx, by)
        if cur is None:
            continue
        if last_possessor is not None and cur != last_possessor:
            passes_made[last_possessor] += 1
            passes_received[cur] += 1
            transfer_log.append((frame_no, last_possessor, cur))
        last_possessor = cur

    print(f"{'tag':10} {'track_id':9} {'votes':30} {'passes_made':12} {'passes_received':16}")
    for tid in sorted(player_tracks, key=lambda x: -passes_made[x]):
        print(f"{display_tag[tid]:10} {tid:9} {str(vote_log[tid]):30} {passes_made[tid]:12} {passes_received[tid]:16}")

    print(f"\n{len(transfer_log)} total transfers logged")
    print("first 15 transfers (frame_no, from, to):")
    for frame_no, a, b in transfer_log[:15]:
        print(f"  frame {frame_no} ({frame_no/29.97:.1f}s): {display_tag[a]} -> {display_tag[b]}")

    # --- visual grid to sanity-check the tags against real crops ---
    crops = []
    for tid in sorted(player_tracks, key=lambda x: display_tag[x]):
        hist = player_tracks[tid]["history"]
        frame_no, (cx, cy) = hist[len(hist) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ret, frame = cap.read()
        if not ret:
            continue
        half_w, half_h = 60, 150
        x1, y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x2, y2 = min(frame.shape[1], int(cx + half_w)), min(frame.shape[0], int(cy) + 10)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (160, 320)) if crop.size else np.zeros((320, 160, 3), np.uint8)
        cv2.putText(crop, display_tag[tid], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(crop, f"made={passes_made[tid]}", (5, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        crops.append(crop)
    rows = []
    for i in range(0, len(crops), 5):
        row = crops[i:i + 5]
        while len(row) < 5:
            row.append(np.zeros((320, 160, 3), np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(OUT_GRID, grid)
    print(f"\nsaved grid to {OUT_GRID}")


if __name__ == "__main__":
    main()
