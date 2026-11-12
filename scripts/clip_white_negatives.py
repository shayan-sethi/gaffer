#!/usr/bin/env python3
"""Hard negatives for a WHITE ball: small round white things that are not the ball (shoes, socks, shorts, shirt flashes) plus every low-confidence
detector hit that is not the confirmed ball. Writes <out_dir>/ball_motion.jsonl in the format ball_ft2.py reads as 'candidates'.
python clip_white_negatives.py <out_dir> <seed_json> [band_y0=380]"""
import sys, json
from pathlib import Path
import numpy as np, cv2
OUT = Path(sys.argv[1]); seeds = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}; Y0 = int(sys.argv[3]) if len(sys.argv) > 3 else 380
meta = json.load(open(OUT / "meta.json")); rows = sorted(json.loads(l)["frame"] for l in open(OUT / "dets.jsonl"))
ball = {json.loads(l)["frame"]: json.loads(l)["cands"] for l in open(OUT / "ball.jsonl")}
import os
MASK = np.load(os.environ["PITCH_MASK"]).astype(np.uint8) if os.environ.get("PITCH_MASK") else None
VERIFIED_FP = [int(v) for v in os.environ.get("VERIFIED_FP_ROWS", "").split("-")] if os.environ.get("VERIFIED_FP_ROWS") else None   # rows (a-b) where low-confidence hits were checked by eye and are NOT the ball
cap = cv2.VideoCapture(meta["video"]); cap.set(1, rows[0]); cur = rows[0]; k3 = np.ones((3, 3), np.uint8); n = 0
with open(OUT / "ball_motion.jsonl", "w") as out:
    for i, fr in enumerate(rows):
        while cur < fr: cap.grab(); cur += 1
        ok, f = cap.read(); cur += 1
        if not ok: break
        if i % 2: continue
        sx = seeds.get(i) or seeds.get(i - 1) or seeds.get(i + 1)
        cs = []
        for c in (ball.get(fr, []) if (VERIFIED_FP and VERIFIED_FP[0] <= i <= VERIFIED_FP[1]) else []):   # only hits I verified are not the ball
            cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
            if sx is None or np.hypot(cx - sx[0], cy - sx[1]) > 25: cs.append([c[0], c[1], c[2], c[3], 0.5, "lowconf"])
        sub = f[Y0:, :]; hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 2] >= 175) & (hsv[..., 1] <= 70)).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        nn, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        for j in range(1, nn):
            x, y, w, h, a = [int(v) for v in st[j]]
            if not (30 <= a <= 420 and 0.5 <= w / max(h, 1) <= 2.0 and a / (w * h) >= 0.5): continue
            cx, cy = x + w / 2, y + h / 2 + Y0
            if MASK is not None and not MASK[int(min(cy + h / 2, MASK.shape[0] - 1)), int(cx)]: continue          # only on the pitch
            if sx is not None and np.hypot(cx - sx[0], cy - sx[1]) <= 25: continue
            ar = w / max(h, 1)
            if 0.7 <= ar <= 1.6 and a <= 260: continue                      # compact + round + ball-sized: could be the ball -> never a negative
            if any(np.hypot(cx - (c[0] + c[2]) / 2, cy - (c[1] + c[3]) / 2) < 45 for c in ball.get(fr, [])): continue          # near any detector hit
            if any(abs(i - k) <= 10 and np.hypot(cx - v[0], cy - v[1]) < 260 for k, v in seeds.items()): continue                  # near the ball's known position in nearby rows
            cs.append([x, y + Y0, x + w, y + h + Y0, 0.4, "white"])
        out.write(json.dumps({"frame": fr, "cands": cs[:8]}) + "\n"); n += len(cs[:8])
print("wrote", n, "negative candidates over", len(rows) // 2, "rows")
