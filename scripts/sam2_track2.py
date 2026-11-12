#!/usr/bin/env python3
"""
SAM 2 per-player tracking WITH detection anchoring, for any fixed-camera clip (no teams, no locker).
 1. seeds = person detections on the frame closest to EXPECT players (env EXPECT) with the least overlap
 2. SAM 2 follows every seed; every CH rows the masks are compared with the person detections:
      - a mask that sits on no detection any more (drifted onto an occluder / background) is pulled back to the nearest unclaimed detection
      - a player whose mask vanished (left the frame, hidden behind someone) is re-attached to an unclaimed detection that looks like him
        (torso colour + size), as soon as one appears
 3. writes identities.json (label N1..Nk, team X = not decided yet), sam2_tracks.json, sam2_track_report.txt
  python sam2_track2.py <out_dir> <band_y1> [min_h=70]
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, cv2, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent)); import device
device.add_sam2_to_path()
from sam2.build_sam import build_sam2_video_predictor

D = Path(sys.argv[1]); Y1 = int(sys.argv[2]); MINH = float(sys.argv[3]) if len(sys.argv) > 3 else 70
EXPECT = int(os.environ.get("EXPECT", 0)); CH = int(os.environ.get("CH", 10)); COLOUR_THR = float(os.environ.get("COLOUR_THR", 40))
meta = json.load(open(D / "meta.json")); Y0 = int(meta["band_y0"]); W = meta["width"]; sfps = meta["fps"] / meta["stride"]
rows = [json.loads(l) for l in open(D / "dets.jsonl")]; N = len(rows)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


MASK = np.load(os.environ["PITCH_MASK"]) if os.environ.get("PITCH_MASK") else None       # drawn pitch zone (feet must be inside)
BIB_SHARE = float(os.environ.get("BIB_SHARE", 0.30))                                       # share of the torso that must be a strong colour (a bib); black / white / grey kit fails
FR = D / "sam2_frames"; FR.mkdir(exist_ok=True); t0 = time.time()
cap = cv2.VideoCapture(meta["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
for i in range(N):
    ok, im = cap.read(); cv2.imwrite(str(FR / f"{i:05d}.jpg"), im[Y0:Y1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    for _ in range(meta["stride"] - 1):
        cap.grab()


class LazyFrames:                                      # frames live on disk (a long clip would not fit in memory); a few recent ones are cached
    def __init__(self):
        self.cache = {}

    def __getitem__(self, i):
        if i not in self.cache:
            if len(self.cache) > 40:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = cv2.imread(str(FR / f"{i:05d}.jpg"))
        return self.cache[i]


frames = LazyFrames()


def bib_share(row, x):
    x1, y1, x2, y2 = [int(v) for v in x[:4]]; h, w = y2 - y1, x2 - x1
    c = frames[row][max(0, y1 - Y0 + int(.2 * h)):max(1, y1 - Y0 + int(.55 * h)), max(0, x1 + int(.2 * w)):max(1, x2 - int(.2 * w))]
    if c.size == 0:
        return 0.0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return float(np.mean((hsv[:, 1] >= 110) & (hsv[:, 2] >= 110)))


def players(row_d, row=None):
    out = []
    for x in row_d:
        if not (x[4] >= 0.5 and x[3] - x[1] >= MINH and Y0 + 5 < x[3] <= Y1 and x[0] > 3 and x[2] < W - 3 and 0.18 <= (x[2] - x[0]) / (x[3] - x[1]) <= 0.8):
            continue
        if MASK is not None and not MASK[min(MASK.shape[0] - 1, int(x[3])), min(MASK.shape[1] - 1, int((x[0] + x[2]) / 2))]:
            continue                                         # feet outside the marked pitch
        if row is not None and bib_share(row, x) < BIB_SHARE:
            continue                                         # not wearing a bib
        out.append(x[:4])
    return drop_contained(out)


def inter_min(a, b):                                   # intersection over the SMALLER box: 1.0 = one box lies completely inside the other
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)


def drop_contained(bs):
    """a box mostly inside a bigger box is a limb / half of the same person, not another player"""
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    keep = []
    for i, b in enumerate(bs):
        if any(j != i and area(bs[j]) > area(b) and inter_min(b, bs[j]) > 0.6 for j in range(len(bs))):
            continue
        keep.append(b)
    return keep


best = None
for r in range(N):
    ps = players(rows[r]["dets"], r); mx = max([inter_min(a, b) for i, a in enumerate(ps) for b in ps[i + 1:]] or [0])
    key = (abs(len(ps) - EXPECT) if EXPECT else -len(ps), mx)
    if best is None or key < best[0]:
        best = (key, r, ps)
SEED, seeds = best[1], sorted(best[2], key=lambda b: b[0]); n = len(seeds)
print(f"seed row {SEED}: {n} players (max overlap {best[0][1]:.2f})", flush=True)
s = frames[SEED].copy()
for k, b in enumerate(seeds):
    cv2.rectangle(s, (int(b[0]), int(b[1] - Y0)), (int(b[2]), int(b[3] - Y0)), (0, 255, 255), 2); cv2.putText(s, str(k + 1), (int(b[0]), int(b[1] - Y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
cv2.imwrite(str(D / "seed.jpg"), s)


def colour(row, b):                                    # torso colour of a box on a given row, Lab (L scaled down: brightness changes a lot with sun / shadow)
    x1, y1, x2, y2 = [int(v) for v in b]; h, w = y2 - y1, x2 - x1
    c = frames[row][max(0, y1 - Y0 + int(.2 * h)):max(1, y1 - Y0 + int(.55 * h)), max(0, x1 + int(.2 * w)):max(1, x2 - int(.2 * w))]
    if c.size == 0:
        return None
    L, a, bb = cv2.cvtColor(np.uint8([[np.median(c.reshape(-1, 3), 0)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([0.3 * L, a - 128, bb - 128])


ref = {oi: colour(SEED, b) for oi, b in enumerate(seeds)}; ref_h = {oi: b[3] - b[1] for oi, b in enumerate(seeds)}
dev = device.pick_device(); print("device:", dev, flush=True)
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", device.sam2_checkpoint(), device=dev)
res, log = {}, []


def collect(fi, oids, logits):
    m = (logits > 0).cpu().numpy()[:, 0]
    for j, oi in enumerate(oids):
        ys, xs = np.where(m[j])
        if len(xs) > 20:
            res.setdefault(fi, {})[int(oi)] = [float(xs.min()), float(ys.min() + Y0), float(xs.max()), float(ys.max() + Y0)]
        elif int(oi) in res.get(fi, {}):
            del res[fi][int(oi)]


with torch.inference_mode(), device.autocast(dev):
    st = pred.init_state(video_path=str(FR), offload_video_to_cpu=True, offload_state_to_cpu=os.environ.get("OFFLOAD_STATE", "1") == "1")   # OFFLOAD_STATE=0 keeps SAM 2's working data on the GPU (faster; needs the memory trimming below)
    for oi, b in enumerate(seeds):
        pred.add_new_points_or_box(st, frame_idx=SEED, obj_id=oi, box=np.array([b[0], b[1] - Y0, b[2], b[3] - Y0], np.float32))
    if SEED > 0:
        for fi, oids, lg in pred.propagate_in_video(st, start_frame_idx=SEED, reverse=True):
            collect(fi, oids, lg)
    r = SEED
    while r < N - 1:
        last = min(N - 1, r + CH)
        for fi, oids, lg in pred.propagate_in_video(st, start_frame_idx=r, max_frame_num_to_track=last - r):
            collect(fi, oids, lg)
        dets = players(rows[last]["dets"], last); now = res.get(last, {}); claimed = set(); fix = {}
        for oi, b in now.items():
            if dets:
                j = max(range(len(dets)), key=lambda j: iou(b, dets[j]))
                if iou(b, dets[j]) >= 0.5:
                    claimed.add(j)
        for oi, b in now.items():                                    # drifted: on no detection
            if dets and max(iou(b, d) for d in dets) < 0.3:
                cand = [(iou(b, dets[j]), j) for j in range(len(dets)) if j not in claimed and iou(b, dets[j]) > 0.05]
                if cand:
                    v, j = max(cand); fix[oi] = dets[j]; claimed.add(j); log.append((last, oi, "drift->detection"))
        for oi in range(n):                                          # lost: no mask on this row
            if oi in now or oi in fix:
                continue
            cand = []
            for j, d in enumerate(dets):
                if j in claimed or any(iou(b, d) > 0.3 for b in now.values()):
                    continue
                c = colour(last, d); h = d[3] - d[1]
                if c is None or ref[oi] is None:
                    continue
                dist = float(np.linalg.norm(c - ref[oi])) + 25 * abs(np.log(h / ref_h[oi]))
                cand.append((dist, j))
            if cand and min(cand)[0] < COLOUR_THR:
                dist, j = min(cand); fix[oi] = dets[j]; claimed.add(j); log.append((last, oi, f"re-attached (dist {dist:.0f})"))
        for oi, d in fix.items():
            pred.add_new_points_or_box(st, frame_idx=last, obj_id=oi, box=np.array([d[0], d[1] - Y0, d[2], d[3] - Y0], np.float32))
        # memory: SAM 2 keeps every past frame's features in RAM (a long clip eventually runs the laptop out of memory); it only ever needs the last few frames
        def _trim(d, keep_from):
            for k_ in [k_ for k_ in d if k_ < keep_from]:
                del d[k_]
        if "output_dict" in st:                              # older SAM 2 versions keep one shared dict as well; this one only keeps one per object
            _trim(st["output_dict"]["non_cond_frame_outputs"], last - 30)
        for od_ in st["output_dict_per_obj"].values():
            _trim(od_["non_cond_frame_outputs"], last - 30)
        if last % 50 < CH:
            print(f"  row {last}/{N} ({time.time() - t0:.0f}s), fixes so far {len(log)}", flush=True)
        r = last
print(f"tracking done ({time.time() - t0:.0f}s); {len(log)} corrections", flush=True)
json.dump({"seed_row": SEED, "tracks": {str(r): {str(o): b for o, b in d.items()} for r, d in res.items()}, "corrections": log}, open(D / "sam2_tracks.json", "w"))
ids = {str(oi): dict(label=f"N{oi + 1}", team="X", role="P", rows={str(r): res[r][oi] + [0] for r in sorted(res) if oi in res[r]}) for oi in range(n)}
json.dump(ids, open(D / "identities.json", "w"))
rep = [f"{n} players seeded on row {SEED}; {N} rows ({N / sfps:.0f} s); {len(log)} corrections ({sum('re-att' in c[2] for c in log)} re-attached, {sum('drift' in c[2] for c in log)} pulled back)"]
for oi in range(n):
    seq = [(r, res[r][oi]) for r in sorted(res) if oi in res[r]]; jumps = 0
    for (r0, b0), (r1, b1) in zip(seq, seq[1:]):
        if r1 - r0 <= 2:
            h = b0[3] - b0[1]; jumps += np.hypot((b0[0] + b0[2]) / 2 - (b1[0] + b1[2]) / 2, b0[3] - b1[3]) > 1.2 * h
    rep.append(f"  N{oi + 1}: tracked on {len(seq)}/{N} rows, teleports {jumps}")
open(D / "sam2_track_report.txt", "w").write("\n".join(rep)); print("\n".join(rep))
