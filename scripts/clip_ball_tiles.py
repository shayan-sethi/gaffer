#!/usr/bin/env python3
"""Ball candidates on very wide frames: the football-ball model on overlapping tiles of the pitch band.
python clip_ball_tiles.py <out_dir> [every=2] [band_y0=470] [band_y1=930] [conf=0.15]   (resumable; writes ball.jsonl)"""
import sys, os, json, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
OUT = Path(sys.argv[1]); EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Y0 = int(sys.argv[3]) if len(sys.argv) > 3 else 470; Y1 = int(sys.argv[4]) if len(sys.argv) > 4 else 930
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
meta = json.load(open(OUT / "meta.json")); xs = meta["tiles"]; TW = 1920
rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
W = meta["width"]
rows_by_frame = {r["frame"]: r["dets"] for r in rows}
frames = [r["frame"] for r in rows][::EVERY]
cache = OUT / "ball.jsonl"
done = {json.loads(l)["frame"] for l in open(cache)} if cache.exists() else set()
todo = [f for f in frames if f not in done]
model = YOLO(os.environ.get("BALL_WEIGHTS", r".local_data\models\football-ball-detection.pt"))
cap = cv2.VideoCapture(meta["video"]); t0, k = time.time(), 0
with open(cache, "a") as out:
    for fi in todo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
        if not ok:
            break
        c = []
        ppl = [d for d in rows_by_frame[fi] if d[4] >= 0.3]            # centre two tiles on the action
        mx = int(np.median([(d[0] + d[2]) / 2 for d in ppl])) if ppl else W // 2
        starts = [min(max(0, mx - 1680), W - TW), min(max(0, mx - 240), W - TW)]
        for x0 in sorted(set(starts)):
            r = model(f[Y0:Y1, x0:x0 + TW], imgsz=TW, conf=CONF, verbose=False)[0]
            for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                c.append([round(float(b[0]) + x0, 1), round(float(b[1]) + Y0, 1), round(float(b[2]) + x0, 1), round(float(b[3]) + Y0, 1), round(float(cf), 3), "ballmodel"])
        out.write(json.dumps({"frame": fi, "cands": c}) + "\n"); out.flush(); k += 1
        if k % 10 == 1:
            el = time.time() - t0; print(f"{k}/{len(todo)} {el/k:.1f}s/frame eta {el/k*(len(todo)-k)/60:.1f} min", flush=True)
print("ball tiles done", flush=True)
