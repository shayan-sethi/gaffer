#!/usr/bin/env python3
"""
Motion-based ball candidates for a FIXED camera (no neural net).

  python clip_motion_ball.py <out_dir> [diff_thresh=28]

Per sampled frame (same frames as dets.jsonl):
  1. foreground = |gray - median background| > thresh (background = per-pixel median of ~60 frames
     spread over the window, so standing players are NOT removed from it but moving things are)
  2. connected components; keep the SMALL, compact ones: 5..150 px area, bbox <= 16 px, roughly square,
     filled (area / bbox area >= 0.45) and not touching a large foreground blob (those are players)
  3. confidence = blob contrast against the surrounding ring
Writes <out_dir>/ball_motion.jsonl: {"frame", "cands": [[x1,y1,x2,y2,conf,"motion"], ...]}.
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2

OUT = Path(sys.argv[1])
THRESH = int(sys.argv[2]) if len(sys.argv) > 2 else 28
meta = json.load(open(OUT / "meta.json"))
rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
S = meta["width"] / 1280.0           # thresholds below are tuned at 720p
Y_MIN, X_MIN = 150 * S, 140 * S         # above this is sky / stands, left of this is the fence


def main():
    cap = cv2.VideoCapture(meta["video"])
    # background: median of 60 frames spread over the window
    pick = np.linspace(0, len(rows) - 1, 60).astype(int)
    stack = []
    for i in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rows[i]["frame"])
        ok, f = cap.read()
        if ok:
            stack.append(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0))
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    cv2.imwrite(str(OUT / "background.jpg"), bg)
    k3 = np.ones((3, 3), np.uint8)
    n_c = []
    with open(OUT / "ball_motion.jsonl", "w") as out:
        for r in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            d = cv2.absdiff(g, bg)
            fg = (d > THRESH).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
            n, lab, st, cen = cv2.connectedComponentsWithStats(fg, connectivity=8)
            big = np.zeros_like(fg)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] > 150 * S * S:
                    big[lab == i] = 1
            big = cv2.dilate(big, np.ones((7, 7), np.uint8))        # a ball touching a player merges: skip those
            for d_ in r["dets"]:                                    # boots / hands / heads of tracked people move too
                if d_[4] >= 0.2:
                    x1, y1, x2, y2 = (int(v) for v in d_[:4])
                    m_ = int(6 * S)
                    big[max(0, y1 - m_):y2 + m_, max(0, x1 - m_):x2 + m_] = 1
            cands = []
            for i in range(1, n):
                x, y, w, h, a = (int(v) for v in st[i])
                if not (5 * S * S <= a <= 150 * S * S and w <= 16 * S and h <= 16 * S and 0.5 <= w / max(h, 1) <= 2.0 and a / (w * h) >= 0.45):
                    continue
                cx, cy = cen[i]
                if cy < (205 * S if cx > 560 * S else Y_MIN) or cx < X_MIN or big[int(cy), int(cx)]:
                    continue
                ring = d[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4]
                inner = d[y:y + h, x:x + w]
                contrast = float(inner[lab[y:y + h, x:x + w] == i].mean())
                ringv = float(ring.mean())
                conf = max(0.0, min(1.0, (contrast - ringv) / 100))
                cands.append([x, y, x + w, y + h, round(conf, 3), "motion"])
            n_c.append(len(cands))
            out.write(json.dumps({"frame": r["frame"], "cands": cands}) + "\n")
    n_c = np.array(n_c)
    print(f"{len(n_c)} frames, candidates/frame: median {np.median(n_c):.0f}, mean {n_c.mean():.1f}, max {n_c.max()}")


if __name__ == "__main__":
    main()
