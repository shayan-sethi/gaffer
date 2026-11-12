#!/usr/bin/env python3
"""
Second detection pass for CROWDED areas (players contesting the ball, bunched at the far end).

Where small players stand close together the first pass returns one merged box or drops a player. This pass finds those groups
(in this frame and the neighbouring sampled frames), crops the region, enlarges it so a player is ~ZOOM_TARGET_H px tall and runs the
person detector again. Zoomed detections replace the first-pass boxes they match; a first-pass box that covers two zoomed people
(a merged box) is dropped.

  python clip_zoom_detect.py <out_dir>
  env: PITCH_MASK=mask.npy  Y1=<bottom row of the picture>  ZOOM_TARGET_H=160  ZOOM_MAX=4  ZOOM_SMALL_H=160

Reads <out_dir>/dets_pass1.jsonl (made from the existing detections on the first run) and writes dets_nolab.jsonl.
Run clip_colours.py afterwards to rebuild dets.jsonl with shirt colours.
"""
import sys, os, json, math, time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from pitch_mask import feet_on_pitch

OUT = Path(sys.argv[1])
meta = json.load(open(OUT / "meta.json"))
W, H = meta["width"], meta["height"]
Y0, Y1 = int(meta.get("band_y0", 0)), int(os.environ.get("Y1", H))
TARGET_H = float(os.environ.get("ZOOM_TARGET_H", 160))
ZMAX = float(os.environ.get("ZOOM_MAX", 4))
SMALL = float(os.environ.get("ZOOM_SMALL_H", 160))
BAND = [float(v) for v in os.environ["ZOOM_BAND"].split(",")] if os.environ.get("ZOOM_BAND") else None   # "y0,y1,scale" in full-frame rows
mask = np.load(os.environ["PITCH_MASK"]).astype(np.uint8) if os.environ.get("PITCH_MASK") else np.ones((H, W), np.uint8)

src = OUT / "dets_pass1.jsonl"
if not src.exists():
    base = OUT / "dets_nolab.jsonl" if (OUT / "dets_nolab.jsonl").exists() else OUT / "dets.jsonl"
    with open(src, "w") as f:
        for l in open(base):
            r = json.loads(l)
            r["dets"] = [d[:8] for d in r["dets"]]
            f.write(json.dumps(r) + "\n")
rows = sorted((json.loads(l) for l in open(src)), key=lambda r: r["frame"])


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


def inside_frac(z, p):
    """share of box z's area that lies inside box p"""
    iw, ih = min(z[2], p[2]) - max(z[0], p[0]), min(z[3], p[3]) - max(z[1], p[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max((z[2] - z[0]) * (z[3] - z[1]), 1e-6)


def groups(dets):
    """regions [x1, y1, x2, y2, median_h] around groups of small on-pitch players standing close together"""
    P = [d for d in dets if d[4] >= 0.2 and 18 <= d[3] - d[1] <= SMALL and feet_on_pitch(mask, d[:4])]
    n = len(P)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = P[i], P[j]
            h = max(a[3] - a[1], b[3] - b[1])
            gapx = max(a[0], b[0]) - min(a[2], b[2])
            if gapx < 0.7 * h and abs(a[3] - b[3]) < 0.8 * h:
                par[find(i)] = find(j)
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(P[i])
    regs = []
    for g in comp.values():
        wide = len(g) == 1 and (g[0][2] - g[0][0]) / max(g[0][3] - g[0][1], 1) > 0.62      # one suspiciously wide box = probably two people
        if len(g) >= 2 or wide:
            hm = float(np.median([d[3] - d[1] for d in g]))
            regs.append([min(d[0] for d in g) - 0.8 * hm, min(d[1] for d in g) - 0.5 * hm,
                         max(d[2] for d in g) + 0.8 * hm, max(d[3] for d in g) + 0.35 * hm, hm])
    return regs


def merge_regions(regs):
    regs = [list(r) for r in regs]
    changed = True
    while changed:
        changed = False
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = regs[i], regs[j]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    regs[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), min(a[4], b[4])]
                    regs.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = []
    for x1, y1, x2, y2, hm in regs:
        out.append([int(max(0, x1)), int(max(Y0, y1)), int(min(W, x2)), int(min(Y1, y2)), hm])
    return [r for r in out if r[2] - r[0] > 20 and r[3] - r[1] > 20]


def main():
    model = YOLO("yolov8n.pt")
    per_frame_regs = [groups(r["dets"]) for r in rows]
    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, rows[0]["frame"])
    cur = rows[0]["frame"]
    t0 = time.time()
    st = dict(regions=0, zoom_dets=0, replaced=0, merged_dropped=0, added=0)
    with open(OUT / "dets_nolab.jsonl", "w") as out:
        for i, r in enumerate(rows):
            while cur < r["frame"]:
                cap.grab()
                cur += 1
            ok, frame = cap.read()
            cur += 1
            if not ok:
                break
            regs = merge_regions([g for k in (i - 1, i, i + 1) if 0 <= k < len(rows) for g in per_frame_regs[k]])
            if BAND:                                         # always look closely at a fixed band (the far end), in overlapping tiles
                by0, by1, bs = BAND
                regs = [[x, int(by0), min(W, x + 600), int(by1), TARGET_H / bs] for x in range(0, W - 100, 500)] +                        [r for r in regs if not (r[1] >= by0 and r[3] <= by1)]
            zoom = []
            for x1, y1, x2, y2, hm in regs:
                s = min(ZMAX, max(1.5, TARGET_H / hm))
                s = min(s, 2400.0 / (x2 - x1), 1600.0 / (y2 - y1))
                if s < 1.25:
                    continue
                up = cv2.resize(frame[y1:y2, x1:x2], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                sz = int(math.ceil(max(up.shape[:2]) / 32) * 32)
                res = model(up, imgsz=sz, conf=0.2, classes=[0], verbose=False)[0]
                st["regions"] += 1
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    # people cut by the crop edge are partial: skip them (unless the crop edge is the picture edge)
                    if (b[0] <= 2 and x1 > 0) or (b[2] >= up.shape[1] - 2 and x2 < W) or (b[1] <= 2 and y1 > Y0) or (b[3] >= up.shape[0] - 2 and y2 < Y1):
                        continue
                    z = [float(b[0]) / s + x1, float(b[1]) / s + y1, float(b[2]) / s + x1, float(b[3]) / s + y1, float(c)]
                    if not BAND and not (0.45 * hm <= z[3] - z[1] <= 2.5 * hm):
                        continue
                    zoom.append(z)
            zoom.sort(key=lambda d: -d[4])
            keepz = []
            for z in zoom:                                   # the same person seen in two overlapping regions
                if all(iou(z, k) < 0.6 for k in keepz):
                    keepz.append(z)
            st["zoom_dets"] += len(keepz)
            final = []
            matched = set()
            for p in r["dets"]:
                cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                in_reg = any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2, _ in regs)
                if in_reg and keepz:
                    best = max(range(len(keepz)), key=lambda k: iou(p, keepz[k]))
                    if iou(p, keepz[best]) >= 0.5:
                        matched.add(best)
                        st["replaced"] += 1
                        continue                              # the zoomed box replaces it
                    if sum(1 for z in keepz if inside_frac(z, p) >= 0.7) >= 2:
                        st["merged_dropped"] += 1
                        continue                              # one first-pass box over two people
                final.append(p[:8])
            for k, z in enumerate(keepz):
                if k not in matched:
                    st["added"] += 1
                final.append([round(z[0], 1), round(z[1], 1), round(z[2], 1), round(z[3], 1), round(z[4], 3), *torso_rgb(frame, z[:4])])
            out.write(json.dumps({"frame": r["frame"], "dets": final}) + "\n")
            if i % 50 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(rows)} {el / (i + 1):.2f}s/frame", flush=True)
    print("zoom pass done:", st, f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
