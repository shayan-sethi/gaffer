#!/usr/bin/env python3
"""Fine-tune the ball detector on SEVERAL clips (e.g. daylight + night). Each source: a clip dir with meta.json, dets.jsonl, ball_motion.jsonl
(colour candidates = hard negatives) and a seed file of ball positions per row (full-frame px) confirmed by eye.
The ball is round: square boxes, no shear/rotation, uniform scale.
python ball_ft2.py <out_dir> <epochs> <clipdir:seedjson:diam> [<clipdir:seedjson:diam> ...]"""
import sys, os, json, random, shutil
from pathlib import Path
import numpy as np, cv2
OUT = Path(sys.argv[1]); EPOCHS = int(sys.argv[2]); SRC = [a.split(",") for a in sys.argv[3:]]
print("sources:", SRC, flush=True)
random.seed(0); np.random.seed(0)
DS = OUT / "ball_ds"; shutil.rmtree(DS, ignore_errors=True)
for sp in ("train", "val"):
    (DS / "images" / sp).mkdir(parents=True); (DS / "labels" / sp).mkdir(parents=True)


def jitter(im, night):
    g = random.uniform(0.7, 1.4) if night else random.uniform(0.8, 1.2)
    return np.clip(im.astype(np.float32) * g + random.uniform(-12, 12), 0, 255).astype(np.uint8)


n_pos = n_neg = 0
for si, (cdir, seedf, diam) in enumerate(SRC):
    cdir = Path(cdir); diam = float(diam); meta = json.load(open(cdir / "meta.json")); Y0 = int(meta.get("band_y0", 0)); W, H = meta["width"], meta["height"]
    rows = sorted(json.loads(l)["frame"] for l in open(cdir / "dets.jsonl"))
    seeds = {int(k): v for k, v in json.load(open(seedf)).items()}
    cands = {}
    if (cdir / "ball_motion.jsonl").exists():
        for l in open(cdir / "ball_motion.jsonl"):
            j = json.loads(l); cands[j["frame"]] = j["cands"]
    last_seed = max(seeds)
    need = set(seeds) | set(range(0, len(rows), 5))
    cap = cv2.VideoCapture(meta["video"]); cap.set(1, rows[0]); cur = rows[0]; frames = {}
    for i, fr in enumerate(rows):
        while cur < fr: cap.grab(); cur += 1
        ok, f = cap.read(); cur += 1
        if not ok: break
        if i in need: frames[i] = f
    night = si > 0

    def crop(f, cx, cy, s):
        win = 640 / s
        x0 = int(np.clip(cx - random.uniform(0.15, 0.85) * win, 0, W - win)); y0 = int(np.clip(cy - random.uniform(0.15, 0.85) * win, Y0, H - win))
        return cv2.resize(f[y0:int(y0 + win), x0:int(x0 + win)], (640, 640), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA), x0, y0

    ks = sorted(seeds)
    for i in ks:
        split = "val" if i in ks[-3:] else "train"
        for v in range(10 if split == "train" else 3):
            s = random.uniform(0.8, 1.3); b = seeds[i]; im, x0, y0 = crop(frames[i], b[0], b[1], s)
            bx, by = (b[0] - x0) * s, (b[1] - y0) * s; d = diam * s
            if random.random() < 0.5: im = im[:, ::-1]; bx = 640 - bx
            name = f"s{si}_pos_{i}_{v}"; cv2.imwrite(str(DS / "images" / split / f"{name}.jpg"), jitter(im, night))
            (DS / "labels" / split / f"{name}.txt").write_text(f"0 {bx / 640:.6f} {by / 640:.6f} {d / 640:.6f} {d / 640:.6f}\n"); n_pos += 1
    for i in sorted(k for k in frames if k not in seeds and all(abs(k - sd) > 3 for sd in seeds)):
        for c in cands.get(rows[i], [])[:2]:
            for v in range(2):
                s = random.uniform(2.4, 3.4); im, x0, y0 = crop(frames[i], (c[0] + c[2]) / 2, (c[1] + c[3]) / 2, s)      # tight window (about 190-260 px): the ball can only be inside it if it is right next to the blob
                if random.random() < 0.5: im = im[:, ::-1]
                name = f"s{si}_neg_{i}_{n_neg}"; cv2.imwrite(str(DS / "images" / "train" / f"{name}.jpg"), jitter(im, night)); (DS / "labels" / "train" / f"{name}.txt").write_text(""); n_neg += 1
print(n_pos, "positive crops,", n_neg, "hard-negative crops", flush=True)
(DS / "data.yaml").write_text(f"path: {DS.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
from ultralytics import YOLO
m = YOLO("yolov8n.pt")
m.train(data=str(DS / "data.yaml"), imgsz=640, epochs=EPOCHS, batch=int(os.environ.get("BATCH", 8)), device=os.environ.get("DEVICE", "cpu"), workers=0, project=str(OUT.resolve()), name="ball_ft2", exist_ok=True,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, degrees=0, shear=0, perspective=0, flipud=0, fliplr=0.0, scale=0.2, mosaic=0.5, close_mosaic=5, patience=15, verbose=False, plots=False)
print("training done", flush=True)
