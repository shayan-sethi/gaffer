#!/usr/bin/env python3
"""
Person detection for very wide frames (e.g. a 6500x1000 panorama): the pitch band is cut into overlapping
tiles that the detector sees at native resolution, then the tile results are merged. Writes the same
dets.jsonl / meta.json that clip_ids.py reads (boxes in full-frame pixels).

  python clip_tiles_detect.py <video> <out_dir> [start_s=0] [seconds=30] [stride=2] [band_y0=420] [tile_w=1920] [overlap=240]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

VIDEO, OUT = sys.argv[1], Path(sys.argv[2])
START = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
Y0 = int(sys.argv[6]) if len(sys.argv) > 6 else 420
TW = int(sys.argv[7]) if len(sys.argv) > 7 else 1920
OV = int(sys.argv[8]) if len(sys.argv) > 8 else 240
import os
Y1 = int(os.environ.get("Y1", 0))                 # optional: bottom of the picture band (e.g. to skip a letterbox)
IMGSZ = int(os.environ.get("IMGSZ", 0))           # optional: detector size (bigger = finds smaller people, slower)


def torso_rgb(frame, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + int(.2 * h), 0):y1 + int(.55 * h), x1 + int(.2 * w):x2 - int(.2 * w)]
    if crop.size == 0:
        return [0, 0, 0]
    m = crop.reshape(-1, 3).mean(0)
    return [round(float(m[2])), round(float(m[1])), round(float(m[0]))]


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    f0 = int(round(START * fps))
    frames = list(range(f0, f0 + int(SECONDS * fps), STRIDE))
    xs = list(range(0, max(1, W - TW + 1), TW - OV))
    if xs[-1] + TW < W:
        xs.append(W - TW)
    (OUT / "meta.json").write_text(json.dumps({"video": VIDEO, "fps": fps, "stride": STRIDE, "start_frame": f0,
                                               "n": 22, "width": W, "height": H, "tiles": xs, "band_y0": Y0}))
    cache = OUT / "dets.jsonl"
    done = {json.loads(l)["frame"] for l in open(cache)} if cache.exists() else set()
    model = YOLO("yolov8n.pt")
    t0, k = time.time(), 0
    todo = [f for f in frames if f not in done]
    with open(cache, "a") as out:
        for fi in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            cand = []
            for x0 in xs:
                tile = frame[Y0:(Y1 or H), x0:x0 + TW]
                r = model(tile, imgsz=(IMGSZ or TW), conf=0.15, classes=[0], verbose=False)[0]
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    cand.append([float(b[0]) + x0, float(b[1]) + Y0, float(b[2]) + x0, float(b[3]) + Y0, float(c)])
            cand.sort(key=lambda d: -d[4])
            keep = []
            for d in cand:                               # merge the tile overlaps
                if all(iou(d, kk) < 0.5 for kk in keep):
                    keep.append(d)
            dets = [[round(d[0], 1), round(d[1], 1), round(d[2], 1), round(d[3], 1), round(d[4], 3), *torso_rgb(frame, d[:4])] for d in keep]
            out.write(json.dumps({"frame": fi, "dets": dets}) + "\n")
            out.flush()
            k += 1
            if k % 10 == 1:
                el = time.time() - t0
                print(f"{k}/{len(todo)} {el / k:.2f}s/frame eta {el / k * (len(todo) - k) / 60:.1f} min", flush=True)
    print("tiles detect done", flush=True)


if __name__ == "__main__":
    main()
