#!/usr/bin/env python3
"""
Try SAM 2 (facebookresearch/sam2, video predictor) as the identity backbone: seed every locked player with his box on the first frame of a short window,
let SAM 2 follow each mask through the window, and compare with the identity locker.
  python sam2_test.py <out_dir> <start_row> <n_rows>        (rows are the 10 fps rows of dets.jsonl; band y0 from meta)
Writes <out_dir>/sam2_test/{frames/,tracks.json,report.txt,sam2_test.webm}.   Weights: .local_data/ext/sam2_ckpt/sam2.1_hiera_small.pt
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, cv2, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent)); import device
device.add_sam2_to_path()
from sam2.build_sam import build_sam2_video_predictor

D = Path(sys.argv[1]); R0 = int(sys.argv[2]); N = int(sys.argv[3])
meta = json.load(open(D / "meta.json")); ids = json.load(open(D / "identities.json"))
Y0 = int(meta.get("band_y0", 0)); Y1 = meta["height"]; W = meta["width"]
OUT = D / "sam2_test"; FR = OUT / "frames"; FR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
cap = cv2.VideoCapture(meta["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"] + R0 * meta["stride"])
frames = []
for i in range(N):
    ok, im = cap.read()
    if not ok:
        N = i; break
    c = im[Y0:Y1]; frames.append(c); cv2.imwrite(str(FR / f"{i:05d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    for _ in range(meta["stride"] - 1):
        cap.grab()
print(f"{N} frames written ({time.time() - t0:.0f}s)", flush=True)

labels = sorted(ids, key=lambda k: ids[k]["label"]); lab = {k: ids[k]["label"] for k in labels}
dev = device.pick_device(); print("device:", dev, flush=True)
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", device.sam2_checkpoint(), device=dev)
res = {}
with torch.inference_mode(), device.autocast(dev):
    st = pred.init_state(video_path=str(FR), offload_video_to_cpu=True, offload_state_to_cpu=True)
    for oi, k in enumerate(labels):
        b = ids[k]["rows"].get(str(R0))
        if b is None:
            print("no start box for", lab[k]); continue
        pred.add_new_points_or_box(st, frame_idx=0, obj_id=oi, box=np.array([b[0], b[1] - Y0, b[2], b[3] - Y0], np.float32))
    print(f"prompted ({time.time() - t0:.0f}s), propagating...", flush=True)
    for fi, oids, logits in pred.propagate_in_video(st):
        m = (logits > 0).cpu().numpy()[:, 0]
        res[fi] = {}
        for j, oi in enumerate(oids):
            ys, xs = np.where(m[j])
            if len(xs) > 20:
                res[fi][int(oi)] = [int(xs.min()), int(ys.min() + Y0), int(xs.max()), int(ys.max() + Y0), int(len(xs))]
        if fi % 20 == 0:
            print(f"  frame {fi}/{N} ({time.time() - t0:.0f}s, GPU mem {device.gpu_mem(dev):.1f} GB)", flush=True)
json.dump({"labels": [lab[k] for k in labels], "tracks": res}, open(OUT / "tracks.json", "w"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


rep = [f"SAM 2 vs locker, rows {R0}-{R0 + N} ({N} frames), seeded from the locker boxes on frame 0"]
for oi, k in enumerate(labels):
    own = other = lost = 0; sw = []
    for fi in range(N):
        t = res.get(fi, {}).get(oi)
        if t is None:
            lost += 1; continue
        best = None; bi = 0
        for k2 in labels:
            b2 = ids[k2]["rows"].get(str(R0 + fi))
            if b2:
                v = iou(t[:4], b2[:4])
                if v > bi:
                    bi, best = v, k2
        if best is None or bi < 0.3:
            continue
        if best == k:
            own += 1
        else:
            other += 1; sw.append((fi, lab[best]))
    rep.append(f"  {lab[k]}: SAM2 mask on the locker's own box {own} frames, on ANOTHER locker identity {other}, no mask {lost}" + ("" if not sw else f"  (first at {sw[0][0] / 10:.1f}s -> {sw[0][1]})"))
open(OUT / "report.txt", "w").write("\n".join(rep)); print("\n".join(rep))
vw = cv2.VideoWriter(str(OUT / "sam2_test.webm"), cv2.VideoWriter_fourcc(*"VP80"), 10, (W * 7 // 10, (Y1 - Y0) * 7 // 10))
cols = [(255, 120, 120), (120, 255, 120), (120, 120, 255), (255, 255, 120), (255, 120, 255), (120, 255, 255), (0, 160, 255), (160, 0, 255), (255, 160, 0), (0, 255, 160), (200, 200, 200), (90, 90, 255)]
for fi in range(N):
    im = frames[fi].copy()
    for oi, t in res.get(fi, {}).items():
        c = cols[oi % 12]; cv2.rectangle(im, (t[0], t[1] - Y0), (t[2], t[3] - Y0), c, 2); cv2.putText(im, lab[labels[oi]], (t[0], max(14, t[1] - Y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    cv2.putText(im, f"SAM 2  t={(R0 + fi) / 10:.1f}s  (labels = locker identities at the first frame)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    vw.write(cv2.resize(im, (W * 7 // 10, (Y1 - Y0) * 7 // 10)))
vw.release(); print(f"done in {time.time() - t0:.0f}s ->", OUT / "sam2_test.webm")
