#!/usr/bin/env python3
"""
Train a ball detector that also handles BIG balls. Dataset from ball_scale_labels.py (labels.json + raw crops): for every label, several training images =
640x640 windows cut from the 1280x720 crop with a random ZOOM (0.9-3.0x, so a 16 px ball becomes up to ~48 px) so the model sees every size, plus
ball-free windows from the same crops as negatives. Fine-tunes the stock football-ball-detection.pt.
  python ball_train_scale.py <labels_dir> <out_dir> [epochs=60] [views=4]      env DEVICE (0 / cpu), MIN_D=14, MIN_CONF=0.6, BIG_D=25, BIG_CONF=0.5
"""
import sys, os, json, random, shutil
from pathlib import Path
import numpy as np, cv2

L = Path(sys.argv[1]); OUT = Path(sys.argv[2]); EP = int(sys.argv[3]) if len(sys.argv) > 3 else 60; VIEWS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MIN_D, MIN_CONF, BIG_D, BIG_CONF = float(os.environ.get("MIN_D", 14)), float(os.environ.get("MIN_CONF", 0.6)), float(os.environ.get("BIG_D", 25)), float(os.environ.get("BIG_CONF", 0.5))
labels = [l for l in json.load(open(L / "labels.json")) if l["diam"] >= MIN_D and l["conf"] >= (BIG_CONF if l["diam"] >= BIG_D else MIN_CONF)]
print(f"{len(labels)} labels kept; diameters: min {min(l['diam'] for l in labels):.0f}, median {np.median([l['diam'] for l in labels]):.0f}, max {max(l['diam'] for l in labels):.0f}", flush=True)
random.seed(3)
shutil.rmtree(OUT / "ds", ignore_errors=True)
for sp in ("train", "val"):
    (OUT / "ds" / "images" / sp).mkdir(parents=True); (OUT / "ds" / "labels" / sp).mkdir(parents=True)
order = sorted(labels, key=lambda l: l["frame"]); val_ids = {l["i"] for l in order[:: 7]}            # every 7th label (spread over the match) is validation
n_tr = n_va = 0
for l in labels:
    im = cv2.imread(str(L / "raw" / f"{l['i']:04d}.jpg")); H, W = im.shape[:2]; x1, y1, x2, y2 = l["box"]; sp = "val" if l["i"] in val_ids else "train"
    for v in range(1 if sp == "val" else VIEWS):
        z = 1.0 if (sp == "val" or v == 0) else float(np.exp(random.uniform(np.log(0.9), np.log(3.0))))       # zoom; the first view is unzoomed
        win = int(round(640 / z)); win = min(win, H)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox = int(np.clip(cx - random.uniform(0.2, 0.8) * win, 0, W - win)); oy = int(np.clip(cy - random.uniform(0.2, 0.8) * win, 0, H - win))
        crop = im[oy:oy + win, ox:ox + win]; crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA); s = 640 / win
        bx1, by1, bx2, by2 = [(x1 - ox) * s, (y1 - oy) * s, (x2 - ox) * s, (y2 - oy) * s]
        if bx1 < 2 or by1 < 2 or bx2 > 638 or by2 > 638:
            continue
        name = f"{l['i']:04d}_{v}"; cv2.imwrite(str(OUT / "ds" / "images" / sp / f"{name}.jpg"), crop)
        open(OUT / "ds" / "labels" / sp / f"{name}.txt", "w").write(f"0 {(bx1 + bx2) / 1280:.6f} {(by1 + by2) / 1280:.6f} {(bx2 - bx1) / 640:.6f} {(by2 - by1) / 640:.6f}\n")
        n_tr += sp == "train"; n_va += sp == "val"
    if sp == "train":                                    # one ball-free window from the same crop (players, turf, boards, goals)
        for _ in range(8):
            win = 640; ox = random.randint(0, W - win); oy = random.randint(0, H - win)
            if not (ox + win < x1 - 40 or ox > x2 + 40 or oy + win < y1 - 40 or oy > y2 + 40):
                continue
            cv2.imwrite(str(OUT / "ds" / "images" / "train" / f"{l['i']:04d}_neg.jpg"), im[oy:oy + win, ox:ox + win]); open(OUT / "ds" / "labels" / "train" / f"{l['i']:04d}_neg.txt", "w").write(""); break
open(OUT / "data.yaml", "w").write(f"path: {(OUT / 'ds').resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
print(f"dataset: {n_tr} train images, {n_va} val images", flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent)); import device
from ultralytics import YOLO
base = os.environ.get("BASE", r".local_data/models/football-ball-detection.pt")
m = YOLO(base)
m.train(data=str(OUT / "data.yaml"), epochs=EP, imgsz=640, batch=16, device=(os.environ.get("DEVICE") or device.yolo_device()), project=str(OUT.resolve()), name="run", exist_ok=True,
        degrees=0, shear=0, perspective=0, flipud=0, fliplr=0.5, mosaic=0.3, scale=0.4, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, patience=20, workers=0, verbose=False, plots=False)
print("weights:", OUT / "run" / "weights" / "best.pt")
