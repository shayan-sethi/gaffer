#!/usr/bin/env python3
"""
Stage 1 for the fixed-camera 5-a-side clip: detect everything on every sampled frame and
cache it (resumable - appends to cache.jsonl as it goes).

Per sampled frame it stores:
  dx      horizontal camera shift vs the first frame (ORB matches on the static top strip)
  dets    [cls, x1, y1, x2, y2, conf, r, g, b]  r,g,b = mean torso colour for non-ball boxes

Usage: python clip5v5_detect.py <video> <out_dir> [start_s=0] [seconds=20] [stride=2]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

VIDEO, OUT = sys.argv[1], Path(sys.argv[2])
START = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
SECONDS = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
MODEL = r".local_data/models/football-player-detection.pt"
CONF = 0.12            # low: ball and far players need it; stage 2 applies per-class thresholds
BALL = 0
STRIP = 300            # rows used for camera-shift estimation (trees / banners / sky)


def torso_rgb(frame, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + int(.2 * h), 0):y1 + int(.55 * h), x1 + int(.2 * w):x2 - int(.2 * w)]
    if crop.size == 0:
        return [0, 0, 0]
    m = crop.reshape(-1, 3).mean(0)          # BGR
    return [round(float(m[2])), round(float(m[1])), round(float(m[0]))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache.jsonl"
    done = set()
    if cache.exists():
        done = {json.loads(l)["frame"] for l in open(cache)}
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = int(round(START * fps))
    frames = list(range(f0, f0 + int(SECONDS * fps), STRIDE))
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, first = cap.read()
    k0, d0 = orb.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)[:STRIP], None)
    last_dx, t0, n = 0.0, time.time(), 0
    (OUT / "meta.json").write_text(json.dumps({"video": VIDEO, "fps": fps, "start_frame": f0, "stride": STRIDE,
                                               "width": int(cap.get(3)), "height": int(cap.get(4))}))
    with open(cache, "a") as out:
        for fi in frames:
            if fi in done:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            k, d = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:STRIP], None)
            if d is not None and d0 is not None:
                m = sorted(bf.match(d0, d), key=lambda x: x.distance)[:200]
                if len(m) >= 20:
                    last_dx = float(np.median([k[x.trainIdx].pt[0] - k0[x.queryIdx].pt[0] for x in m]))
            r = model(frame, imgsz=1280, verbose=False, conf=CONF)[0]
            dets = []
            for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                col = [0, 0, 0] if int(c) == BALL else torso_rgb(frame, b)
                dets.append([int(c), *[round(float(v), 1) for v in b], round(float(cf), 3), *col])
            out.write(json.dumps({"frame": fi, "dx": round(last_dx, 1), "dets": dets}) + "\n")
            out.flush()
            n += 1
            if n % 10 == 1:
                el = time.time() - t0
                print(f"{n}/{len(frames) - len(done)} {el / n:.2f}s/frame eta {el / n * (len(frames) - len(done) - n) / 60:.1f} min",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
