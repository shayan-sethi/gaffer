#!/usr/bin/env python3
"""
Ball candidates for a clip already processed by clip_ids.py (same <out_dir>, same sampled frames).
Runs the COCO 'sports ball' class and the dedicated football-ball model on every `every`-th sampled
frame (default 2 -> 7.5 fps) and appends to <out_dir>/ball.jsonl (resumable).

  python clip_ball.py <out_dir> [every=2]
"""
import sys, json, time
from pathlib import Path
import cv2
from ultralytics import YOLO

OUT = Path(sys.argv[1])
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2
meta = json.load(open(OUT / "meta.json"))
rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
frames = [r["frame"] for r in rows][::EVERY]
cache = OUT / "ball.jsonl"
done = {json.loads(l)["frame"] for l in open(cache)} if cache.exists() else set()
todo = [f for f in frames if f not in done]
coco = YOLO("yolov8n.pt")
ballm = YOLO(r".local_data\models\football-ball-detection.pt")
cap = cv2.VideoCapture(meta["video"])
t0, k = time.time(), 0
with open(cache, "a") as out:
    for fi in todo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        c = []
        r = coco(frame, imgsz=1280, conf=0.05, classes=[32], verbose=False)[0]
        for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            c.append([*[round(float(v), 1) for v in b], round(float(cf), 3), "coco"])
        r = ballm(frame, imgsz=1280, conf=0.08, verbose=False)[0]
        for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            c.append([*[round(float(v), 1) for v in b], round(float(cf), 3), "ballmodel"])
        out.write(json.dumps({"frame": fi, "cands": c}) + "\n")
        out.flush()
        k += 1
        if k % 10 == 1:
            el = time.time() - t0
            print(f"{k}/{len(todo)} {el / k:.2f}s/frame eta {el / k * (len(todo) - k) / 60:.1f} min", flush=True)
print("ball done", flush=True)
