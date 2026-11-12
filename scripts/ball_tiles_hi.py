#!/usr/bin/env python3
"""Ball candidates at a HIGH frame rate (every `step` video frames, e.g. 2 = 25 fps on a 50 fps clip) so fast balls are not lost between samples.
Writes <out_dir>/ball_hi.jsonl  ({"frame": video_frame, "cands": [[x1,y1,x2,y2,conf,"ballmodel"],...]}); resumable.
python ball_tiles_hi.py <out_dir> [step=2] [band_y0=380] [band_y1=1080] [conf=0.08]     env BALL_WEIGHTS"""
import sys, os, json, time
from pathlib import Path
import cv2
from ultralytics import YOLO
OUT = Path(sys.argv[1]); STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08
meta = json.load(open(OUT / "meta.json"))
Y0 = int(sys.argv[3]) if len(sys.argv) > 3 else int(meta.get("band_y0", 0)); Y1 = int(sys.argv[4]) if len(sys.argv) > 4 else int(meta["height"])   # defaults come from the clip, nothing is hard-coded per feed
WID = int(meta["width"])
rows = sorted((json.loads(l) for l in open(OUT / "dets.jsonl")), key=lambda r: r["frame"])
first, last = rows[0]["frame"], rows[-1]["frame"]
frames = list(range(first, last + 1, STEP))
cache = OUT / "ball_hi.jsonl"
done = {json.loads(l)["frame"] for l in open(cache)} if cache.exists() else set()
todo = [f for f in frames if f not in done]
model = YOLO(os.environ.get("BALL_WEIGHTS", r".local_data/models/football-ball-detection.pt"))
TILE_W = int(os.environ.get("TILE_W", 0))      # >0: cut the band into native-resolution tiles of this width (a small ball stays full size); 0: whole width, shrunk to <=1920


def detect(f):
    xs = [0] if not TILE_W or WID <= TILE_W else sorted(set(list(range(0, WID - TILE_W + 1, TILE_W - 128)) + [WID - TILE_W]))
    tw = WID if not TILE_W else min(TILE_W, WID); out = []
    for x0 in xs:
        r = model(f[Y0:Y1, x0:x0 + tw], imgsz=min(tw, 1920) if not TILE_W else tw, conf=CONF, verbose=False)[0]
        for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append([round(float(b[0]) + x0, 1), round(float(b[1]) + Y0, 1), round(float(b[2]) + x0, 1), round(float(b[3]) + Y0, 1), round(float(cf), 3), "ballmodel"])
    out.sort(key=lambda c: -c[4]); keep = []
    for c in out:                                                  # tiles overlap: keep the best of candidates that coincide
        if all(abs((c[0] + c[2]) / 2 - (k[0] + k[2]) / 2) > 8 or abs((c[1] + c[3]) / 2 - (k[1] + k[3]) / 2) > 8 for k in keep):
            keep.append(c)
    return keep


cap = cv2.VideoCapture(meta["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, todo[0] if todo else 0)
pos = todo[0] if todo else 0; t0, k = time.time(), 0
with open(cache, "a") as out:
    for fi in todo:
        while pos < fi:                      # sequential read: grab() is far cheaper than seeking
            cap.grab(); pos += 1
        ok, f = cap.read(); pos += 1
        if not ok:
            break
        c = detect(f)
        out.write(json.dumps({"frame": fi, "cands": c}) + "\n"); out.flush(); k += 1
        if k % 50 == 1:
            el = time.time() - t0; print(f"{k}/{len(todo)} {el/k:.2f}s/frame eta {el/k*(len(todo)-k)/60:.1f} min", flush=True)
print("ball hi done", flush=True)
