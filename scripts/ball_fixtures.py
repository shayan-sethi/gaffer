#!/usr/bin/env python3
"""
Find FIXTURES: spots where the ball detector keeps firing in the same place across the whole match (a bag by the goal, a mark on the boards, a cone).
Samples N frames spread over the whole video, runs the ball detector on each, and keeps the spots that appear in >= share of the samples.
A real ball is only ever at one place for a few seconds, so it can never reach that share; a ball lying still stays unaffected.
  python ball_fixtures.py <video> <out_json> <band_y0> <band_y1> [samples=40] [share=0.25]      env BALL_WEIGHTS, TILE_W
"""
import sys, os, json
import numpy as np, cv2
from ultralytics import YOLO

V, OUT, Y0, Y1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 40; SHARE = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
TILE_W = int(os.environ.get("TILE_W", 0))
model = YOLO(os.environ.get("BALL_WEIGHTS", r".local_data\models\football-ball-detection.pt"))
cap = cv2.VideoCapture(V); n = int(cap.get(7)); W = int(cap.get(3))
xs = [0] if not TILE_W or W <= TILE_W else sorted(set(list(range(0, W - TILE_W + 1, TILE_W - 128)) + [W - TILE_W])); tw = W if not TILE_W else min(TILE_W, W)
pts = []
for k, fi in enumerate(np.linspace(int(n * 0.02), int(n * 0.98), NS).astype(int)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
    if not ok:
        continue
    seen = []
    for x0 in xs:
        r = model(f[Y0:Y1, x0:x0 + tw], imgsz=min(tw, 1920) if not TILE_W else tw, conf=0.15, verbose=False)[0]
        for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            seen.append(((b[0] + b[2]) / 2 + x0, (b[1] + b[3]) / 2 + Y0))
    pts.append(seen)
used = len(pts); clusters = []                                   # [x, y, set of sample indices]
for i, seen in enumerate(pts):
    for (x, y) in seen:
        for c in clusters:
            if abs(c[0] - x) <= 8 and abs(c[1] - y) <= 8:
                c[2].add(i); c[0] = 0.8 * c[0] + 0.2 * x; c[1] = 0.8 * c[1] + 0.2 * y; break
        else:
            clusters.append([x, y, {i}])
fix = [dict(x=round(c[0], 1), y=round(c[1], 1), r=10, share=round(len(c[2]) / used, 2)) for c in clusters if len(c[2]) / used >= SHARE]
json.dump(fix, open(OUT, "w"), indent=1)
top = sorted(clusters, key=lambda c: -len(c[2]))[:6]
print("most frequent spots (x, y, share of samples):", [(round(c[0]), round(c[1]), round(len(c[2]) / used, 2)) for c in top])
print(f"{used} sampled frames, {len(clusters)} spots, {len(fix)} fixtures:", fix)
