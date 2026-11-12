#!/usr/bin/env python3
"""
Fine-tune a small ball detector on THIS venue's ball (e.g. a red ball on green turf).
Seeds: rows 0..N_SEED of a clip where the ball position was confirmed by eye (ball_path.json); hard negatives: crops around red
candidates (boots, kit) elsewhere in the clip. The ball is ROUND: boxes are square, and the augmentations keep it round (no
shear/rotation that squashes it, scale kept uniform).
  python ball_ft.py <out_dir> [n_seed=29] [epochs=40]
Result: <out_dir>/ball_ft/weights/best.pt
"""
import sys, os, json, random, shutil
from pathlib import Path
import numpy as np, cv2
OUT = Path(sys.argv[1]); N_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 29
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
meta = json.load(open(OUT / "meta.json")); Y0 = int(meta.get("band_y0", 0))
rows = sorted(json.loads(l)["frame"] for l in open(OUT / "dets.jsonl"))
ball = {int(k): v for k, v in json.load(open(OUT / "ball_path.json")).items()}
cands = {}
for l in open(OUT / "ball_motion.jsonl"):
    j = json.loads(l); cands[j["frame"]] = j["cands"]
random.seed(0); np.random.seed(0)
DS = OUT / "ball_ds"; shutil.rmtree(DS, ignore_errors=True)
for sp in ("train", "val"):
    (DS / "images" / sp).mkdir(parents=True); (DS / "labels" / sp).mkdir(parents=True)
BALL_D = 14.0                                          # ball diameter in px at native size (measured on the seeds: 12-14)
cap = cv2.VideoCapture(meta["video"]); cap.set(1, rows[0]); cur = rows[0]
frames = {}
need = set(range(N_SEED)) | set(range(N_SEED, len(rows), 7))
for i, fr in enumerate(rows):
    while cur < fr: cap.grab(); cur += 1
    ok, f = cap.read(); cur += 1
    if not ok: break
    if i in need: frames[i] = f
print(len(frames), "frames loaded")
H, W = meta["height"], meta["width"]


def crop(f, cx, cy, s):
    """640x640 output; source window of size 640/s around (cx,cy) shifted randomly; returns img, (offset x, y, scale)"""
    win = 640 / s
    x0 = int(np.clip(cx - random.uniform(0.15, 0.85) * win, 0, W - win)); y0 = int(np.clip(cy - random.uniform(0.15, 0.85) * win, Y0, H - win))
    im = cv2.resize(f[y0:int(y0 + win), x0:int(x0 + win)], (640, 640), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return im, x0, y0


def jitter(im):
    im = im.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-15, 15)
    return np.clip(im, 0, 255).astype(np.uint8)


n_pos = 0
for i in range(N_SEED):
    b = ball.get(i)
    if b is None: continue
    split = "val" if i >= N_SEED - 4 else "train"
    for v in range(10 if split == "train" else 3):
        s = random.uniform(0.8, 1.3); im, x0, y0 = crop(frames[i], b[0], b[1], s); win = 640 / s
        bx, by = (b[0] - x0) * s, (b[1] - y0) * s; d = BALL_D * s
        if random.random() < 0.5: im = im[:, ::-1]; bx = 640 - bx
        name = f"pos_{i}_{v}"
        cv2.imwrite(str(DS / "images" / split / f"{name}.jpg"), jitter(im))
        (DS / "labels" / split / f"{name}.txt").write_text(f"0 {bx / 640:.6f} {by / 640:.6f} {d / 640:.6f} {d / 640:.6f}\n"); n_pos += 1
n_neg = 0
for i in sorted(k for k in frames if k >= N_SEED):
    cs = cands.get(rows[i], [])
    for c in cs[:2]:                                       # crops centred on red false candidates (boots, kit, skin)
        for v in range(2):
            s = random.uniform(0.8, 1.3); im, x0, y0 = crop(frames[i], (c[0] + c[2]) / 2, (c[1] + c[3]) / 2, s)
            if random.random() < 0.5: im = im[:, ::-1]
            name = f"neg_{i}_{n_neg}"; cv2.imwrite(str(DS / "images" / "train" / f"{name}.jpg"), jitter(im))
            (DS / "labels" / "train" / f"{name}.txt").write_text(""); n_neg += 1
print(n_pos, "positive crops,", n_neg, "hard-negative crops")
(DS / "data.yaml").write_text(f"path: {DS.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
from ultralytics import YOLO
m = YOLO("yolov8n.pt")
m.train(data=str(DS / "data.yaml"), imgsz=640, epochs=EPOCHS, batch=8, device="cpu", workers=0, project=str(OUT.resolve()), name="ball_ft", exist_ok=True,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, degrees=0, shear=0, perspective=0, flipud=0, fliplr=0.0, scale=0.2, mosaic=0.5, close_mosaic=5, patience=15, verbose=False, plots=False)
print("training done", flush=True)
