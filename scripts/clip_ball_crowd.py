#!/usr/bin/env python3
"""Find the ball INSIDE crowds. Where players bunch up, the ball is among their feet. For each sampled row: take the identities (identities.json),
group players standing close together, zoom into each group (and every lone player's feet), and look for a small round ball-coloured blob
at foot height. Armbands / skin / bibs are rejected because they are not at foot height.
python clip_ball_crowd.py <out_dir> [every=2]   env: BALL_HSV="hmax,hmin,smin,vmin"  ZOOM=4   -> writes ball_motion.jsonl (colour candidates, feet-filtered)"""
import sys, os, json
from pathlib import Path
import numpy as np, cv2
OUT = Path(sys.argv[1]); EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2
meta = json.load(open(OUT / "meta.json")); ids = json.load(open(OUT / "identities.json"))
rows = sorted(json.loads(l)["frame"] for l in open(OUT / "dets.jsonl"))
hmax, hmin, smin, vmin = [int(v) for v in os.environ.get("BALL_HSV", "18,158,45,70").split(",")]; Z = float(os.environ.get("ZOOM", 4))
cap = cv2.VideoCapture(meta["video"]); cap.set(1, rows[0]); cur = rows[0]
k3 = np.ones((3, 3), np.uint8); n_rows = n_c = 0
with open(OUT / "ball_motion.jsonl", "w") as out:
    for i, fr in enumerate(rows):
        while cur < fr: cap.grab(); cur += 1
        ok, f = cap.read(); cur += 1
        if not ok: break
        if i % EVERY: continue
        P = []
        for I in ids.values():
            b = I["rows"].get(str(i))
            if b and I["role"] != "REF": P.append(b[:4])
        # zones where the ball can be 'in the feet': around each player's boots (wider for crowds - the ball is between the legs)
        zones = [(b[0] - 0.9 * (b[2] - b[0]) - 6, b[3] - 0.38 * (b[3] - b[1]), b[2] + 0.9 * (b[2] - b[0]) + 6, b[3] + 0.3 * (b[3] - b[1])) for b in P]
        cands = []
        for x1, y1, x2, y2 in zones:
            x1, y1, x2, y2 = int(max(0, x1)), int(max(204, y1)), int(min(f.shape[1], x2)), int(min(f.shape[0], y2))
            if x2 - x1 < 8 or y2 - y1 < 8: continue
            crop = cv2.resize(f[y1:y2, x1:x2], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC); hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            m = (((hsv[..., 0] <= hmax) | (hsv[..., 0] >= hmin)) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3 * 1)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
            for j in range(1, n):
                x, y, w, h, a = [int(v) for v in st[j]]
                if not (60 <= a <= 160 * Z * Z * 0.5 and 0.6 <= w / max(h, 1) <= 1.7 and a / (w * h) >= 0.6 and w <= 22 * Z and h <= 22 * Z): continue
                cands.append([x1 + x / Z, y1 + y / Z, x1 + (x + w) / Z, y1 + (y + h) / Z, round(min(1.0, 0.4 + 0.6 * a / (w * h)), 3), "crowd"])
        # merge duplicates from overlapping zones
        keep = []
        for c in sorted(cands, key=lambda c: -c[4]):
            if all(np.hypot((c[0] + c[2]) / 2 - (k[0] + k[2]) / 2, (c[1] + c[3]) / 2 - (k[1] + k[3]) / 2) > 5 for k in keep): keep.append(c)
        out.write(json.dumps({"frame": fr, "cands": keep}) + "\n"); n_rows += 1; n_c += len(keep)
print(f"{n_rows} rows, {n_c/max(n_rows,1):.1f} candidates per row", flush=True)
