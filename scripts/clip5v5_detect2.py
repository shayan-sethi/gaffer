#!/usr/bin/env python3
"""
Stage 1b: better detectors for the fixed 5-a-side clip. Reuses frames + camera shift (dx) and the
football-model ball candidates from cache.jsonl (clip5v5_detect.py) and adds:
  - people:  COCO yolov8n 'person' (finds the dark-kit players the football model misses)
  - ball:    dedicated football-ball-detection.pt + COCO 'sports ball' (class 32)
Writes cache2.jsonl in the same format (cls 2 = person, cls 0 = ball candidate).

Usage: python clip5v5_detect2.py <out_dir>
"""
import sys, json, time
from pathlib import Path
import cv2
from ultralytics import YOLO

OUT = Path(sys.argv[1])
meta = json.load(open(OUT / "meta.json"))
base = [json.loads(l) for l in open(OUT / "cache.jsonl")]
base.sort(key=lambda r: r["frame"])
done = set()
if (OUT / "cache2.jsonl").exists():
    done = {json.loads(l)["frame"] for l in open(OUT / "cache2.jsonl")}


def torso_rgb(frame, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + int(.2 * h), 0):y1 + int(.55 * h), x1 + int(.2 * w):x2 - int(.2 * w)]
    if crop.size == 0:
        return [0, 0, 0]
    m = crop.reshape(-1, 3).mean(0)
    return [round(float(m[2])), round(float(m[1])), round(float(m[0]))]


def main():
    coco = YOLO("yolov8n.pt")
    ballm = YOLO(r".local_data\models\football-ball-detection.pt")
    cap = cv2.VideoCapture(meta["video"])
    t0, n = time.time(), 0
    todo = [r for r in base if r["frame"] not in done]
    with open(OUT / "cache2.jsonl", "a") as out:
        for r in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, frame = cap.read()
            if not ok:
                break
            dets = []
            res = coco(frame, imgsz=1280, conf=0.05, classes=[0, 32], verbose=False)[0]
            for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                if int(c) == 0 and cf >= 0.2:
                    dets.append([2, *[round(float(v), 1) for v in b], round(float(cf), 3), *torso_rgb(frame, b)])
                elif int(c) == 32:
                    dets.append([0, *[round(float(v), 1) for v in b], round(float(cf), 3), 0, 0, 0])
            res = ballm(frame, imgsz=1280, conf=0.08, verbose=False)[0]
            for b, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                dets.append([0, *[round(float(v), 1) for v in b], round(float(cf), 3), 0, 0, 0])
            dets += [d for d in r["dets"] if d[0] == 0]          # football-model ball candidates
            out.write(json.dumps({"frame": r["frame"], "dx": r["dx"], "dets": dets}) + "\n")
            out.flush()
            n += 1
            if n % 10 == 1:
                el = time.time() - t0
                print(f"{n}/{len(todo)} {el / n:.2f}s/frame eta {el / n * (len(todo) - n) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
