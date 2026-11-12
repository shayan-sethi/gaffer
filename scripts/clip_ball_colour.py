#!/usr/bin/env python3
"""Ball candidates by COLOUR (for a visibly coloured ball on a fixed camera, e.g. a red ball on green turf).
Red/orange round blobs of the right size inside the pitch region. Writes <out_dir>/ball_motion.jsonl (same format the pass script reads).
python clip_ball_colour.py <out_dir> [every=2]     env: PITCH_MASK, BALL_HSV="h1,h2,smin,vmin"   (hue range h1..h2 on the red side, wrapping through 0)
"""
import sys, os, json
from pathlib import Path
import numpy as np, cv2
OUT = Path(sys.argv[1]); EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2
meta = json.load(open(OUT / "meta.json")); S = meta["width"] / 1280.0
rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
mask = np.load(os.environ["PITCH_MASK"]).astype(np.uint8) if os.environ.get("PITCH_MASK") else np.ones((meta["height"], meta["width"]), np.uint8)
hmax, hmin, smin, vmin = [int(v) for v in os.environ.get("BALL_HSV", "12,168,110,110").split(",")]
cap = cv2.VideoCapture(meta["video"]); cap.set(1, rows[0]["frame"]); cur = rows[0]["frame"]
k3 = np.ones((3, 3), np.uint8); n_c = []
with open(OUT / "ball_motion.jsonl", "w") as out:
    for i, r in enumerate(rows):
        while cur < r["frame"]:
            cap.grab(); cur += 1
        ok, f = cap.read(); cur += 1
        if not ok: break
        if i % EVERY: continue
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        red = (((hsv[..., 0] <= hmax) | (hsv[..., 0] >= hmin)) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)).astype(np.uint8) * mask
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k3)
        n, lab, st, cen = cv2.connectedComponentsWithStats(red, connectivity=8)
        cands = []
        for j in range(1, n):
            x, y, w, h, a = [int(v) for v in st[j]]
            if not (float(os.environ.get("BALL_MINAREA", 12)) * S * S <= a <= 700 * S * S and w <= 40 * S and h <= 40 * S and 0.5 <= w / max(h, 1) <= 2.0 and a / (w * h) >= 0.5): continue
            cands.append([x, y, x + w, y + h, round(min(1.0, 0.4 + a / (w * h) * 0.5), 3), "colour"])
        n_c.append(len(cands)); out.write(json.dumps({"frame": r["frame"], "cands": cands}) + "\n")
n_c = np.array(n_c); print(f"{len(n_c)} rows, candidates per row: median {np.median(n_c):.0f}, mean {n_c.mean():.1f}, rows with none {int((n_c == 0).sum())}")
