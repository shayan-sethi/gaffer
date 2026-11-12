#!/usr/bin/env python3
"""Ball detection on ENLARGED tiles of a band of the picture (for a small, far ball).
python clip_ball_zoom.py <out_dir> <row0> <row1> [every=2]    env: BALL_WEIGHTS, ZB="y0,y1,scale" (full-frame rows), ZTILE=640
Appends to <out_dir>/ball_zoom.jsonl ({"frame","cands":[[x1,y1,x2,y2,conf,"zoom"],...]})"""
import sys, os, json, time
from pathlib import Path
import cv2
from ultralytics import YOLO
OUT = Path(sys.argv[1]); R0, R1 = int(sys.argv[2]), int(sys.argv[3]); EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 2
y0, y1, SC = [float(v) for v in os.environ.get("ZB", "480,650,3").split(",")]; y0, y1 = int(y0), int(y1); TW = int(os.environ.get("ZTILE", 640))
meta = json.load(open(OUT / "meta.json")); W = meta["width"]
rows = sorted(json.loads(l)["frame"] for l in open(OUT / "dets.jsonl"))
model = YOLO(os.environ.get("BALL_WEIGHTS", ".local_data/both_ball/ball_ft2/weights/best.pt"))
cap = cv2.VideoCapture(meta["video"]); cache = OUT / "ball_zoom.jsonl"; t0 = time.time(); n = 0
done = {json.loads(l)["frame"] for l in open(cache)} if cache.exists() else set()
with open(cache, "a") as out:
    for i in range(R0, R1 + 1, EVERY):
        fr = rows[i]
        if fr in done: continue
        cap.set(1, fr); ok, f = cap.read()
        if not ok: break
        c = []
        for x0 in range(0, W - 100, TW - 80):
            x1 = min(W, x0 + TW); tile = cv2.resize(f[y0:y1, x0:x1], None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
            r = model(tile, imgsz=int(-(-max(tile.shape[:2]) // 32) * 32), conf=0.1, verbose=False)[0]
            for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                c.append([round(float(b[0]) / SC + x0, 1), round(float(b[1]) / SC + y0, 1), round(float(b[2]) / SC + x0, 1), round(float(b[3]) / SC + y0, 1), round(float(cf), 3), "zoom"])
        out.write(json.dumps({"frame": fr, "cands": c}) + "\n"); out.flush(); n += 1
print(f"{n} frames in {time.time()-t0:.0f}s", flush=True)
