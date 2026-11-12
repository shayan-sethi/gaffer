#!/usr/bin/env python3
"""
Build labels for BIG balls (a ball near the camera looks several times larger than the model was trained on).
Samples frames spread over a match (excluding a held-out window), runs the existing ball detector at several scales of the pitch region
(1.0 native tiles, 0.5 and 0.33 shrunk copies, so a large ball becomes the size the model knows), keeps confident detections, maps them back to full
resolution, and saves: raw crops with the label, plus a contact sheet to check by eye.
  python ball_scale_labels.py <video> <out_dir> <y0> <y1> [n_frames=300] [hold_start_s hold_end_s]      env BALL_WEIGHTS
"""
import sys, os, json, random
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

V, OUT = sys.argv[1], Path(sys.argv[2]); Y0, Y1 = int(sys.argv[3]), int(sys.argv[4])
NF = int(sys.argv[5]) if len(sys.argv) > 5 else 300
HOLD = (float(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 7 else (-1, -1)
CONF = float(os.environ.get("CONF", 0.45))
(OUT / "raw").mkdir(parents=True, exist_ok=True)
model = YOLO(os.environ.get("BALL_WEIGHTS", r".local_data/models/football-ball-detection.pt"))
cap = cv2.VideoCapture(V); fps = cap.get(5); n = int(cap.get(7)); W = int(cap.get(3))
random.seed(7)
cand = [f for f in range(int(fps * 120), int(n - fps * 30), 1) if not (HOLD[0] * fps <= f <= HOLD[1] * fps)]
frames = sorted(random.sample(cand, NF)); labels = []; sheet = []


def run(img, imgsz, conf=CONF):
    r = model(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    return [(b[0], b[1], b[2], b[3], float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


for k, fi in enumerate(frames):
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
    except SystemError:                                  # the decoder occasionally fails on a seek: reopen and try once more, else skip the frame
        cap.release(); cap = cv2.VideoCapture(V)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
        except SystemError:
            continue
    if not ok:
        continue
    reg = f[Y0:Y1]; hits = []
    for x0 in sorted(set(list(range(0, W - 1280 + 1, 1152)) + [W - 1280])):                      # scale 1.0: native tiles
        for b in run(reg[:, x0:x0 + 1280], 1280):
            hits.append((b[0] + x0, b[1] + Y0, b[2] + x0, b[3] + Y0, b[4], 1.0))
    for s in (0.5, 0.33):                                                                          # shrunk copies: a big ball becomes a normal-sized one
        sm = cv2.resize(reg, (int(reg.shape[1] * s), int(reg.shape[0] * s)))
        for b in run(sm, max(sm.shape[:2])):
            hits.append((b[0] / s, b[1] / s + Y0, b[2] / s, b[3] / s + Y0, b[4], s))
    hits.sort(key=lambda h: -h[4]); keep = []
    for h in hits:                                                                                 # merge duplicates across scales
        if all(abs((h[0] + h[2]) / 2 - (q[0] + q[2]) / 2) > 0.6 * max(h[2] - h[0], q[2] - q[0]) or abs((h[1] + h[3]) / 2 - (q[1] + q[3]) / 2) > 0.6 * max(h[3] - h[1], q[3] - q[1]) for q in keep):
            keep.append(h)
    for h in keep[:1]:                                                                             # at most the best one per frame
        d = h[2] - h[0]
        if not (6 <= d <= 120) or abs((h[3] - h[1]) - d) > 0.5 * d:                                # round-ish and a plausible size
            continue
        cx, cy = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2; x0 = int(min(max(0, cx - 640), W - 1280)); y0 = int(min(max(0, cy - 360), f.shape[0] - 720))
        crop = f[y0:y0 + 720, x0:x0 + 1280]; i = len(labels)
        cv2.imwrite(str(OUT / "raw" / f"{i:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
        labels.append(dict(i=i, frame=int(fi), t=round(float(fi / fps), 1), box=[round(float(h[0] - x0), 1), round(float(h[1] - y0), 1), round(float(h[2] - x0), 1), round(float(h[3] - y0), 1)], diam=round(float(d), 1), conf=round(float(h[4]), 2), scale=float(h[5]), crop=[int(x0), int(y0)]))
        sheet.append((d, crop[int(max(0, cy - y0 - 60)):int(cy - y0 + 60), int(max(0, cx - x0 - 60)):int(cx - x0 + 60)].copy(), h[4], h[5]))
    if k % 50 == 0:
        print(f"{k}/{NF} frames, {len(labels)} labels", flush=True)
json.dump(labels, open(OUT / "labels.json", "w"), indent=1)
sheet.sort(key=lambda s: -s[0]); tiles = []
for d, c, cf, sc in sheet[:96]:
    c = cv2.resize(c, (120, 120)) if c.size else np.zeros((120, 120, 3), np.uint8)
    cv2.putText(c, f"{d:.0f}px {cf:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1); tiles.append(c)
while len(tiles) % 12:
    tiles.append(np.zeros((120, 120, 3), np.uint8))
if tiles:
    cv2.imwrite(str(OUT / "sheet.jpg"), np.vstack([np.hstack(tiles[r:r + 12]) for r in range(0, len(tiles), 12)]))
ds = np.array([l["diam"] for l in labels]) if labels else np.zeros(1)
print(f"{len(labels)} labels from {NF} frames; ball diameter px: min {ds.min():.0f}, median {np.median(ds):.0f}, max {ds.max():.0f}; >=25px: {int((ds >= 25).sum())}")
