#!/usr/bin/env python3
"""
Per-player tracking with SAM 2 on any fixed-camera clip, NO team logic and no locker: seeds = person detections on the frame where the most players
are visible and apart, SAM 2 follows each one forward and backward. Then a switch check that needs no ground truth:
  * colour class of the torso (orange / yellow-green / other bibs) per object per row -> a flip means the mask moved to someone else
  * centre jumps bigger than 1.2 body heights between consecutive rows (teleports)
  python sam2_track.py <out_dir> <band_y1> [min_h=70]         (needs dets.jsonl + meta.json from clip_tiles_detect.py; band_y0 from meta)
Writes sam2_tracks.json, sam2_track_report.txt, seed.jpg, sam2_track.webm
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, cv2, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent)); import device
device.add_sam2_to_path()
from sam2.build_sam import build_sam2_video_predictor

D = Path(sys.argv[1]); Y1 = int(sys.argv[2]); MINH = float(sys.argv[3]) if len(sys.argv) > 3 else 70
EXPECT = int(os.environ.get("EXPECT", 0))
meta = json.load(open(D / "meta.json")); Y0 = int(meta["band_y0"]); W = meta["width"]; sfps = meta["fps"] / meta["stride"]
rows = [json.loads(l) for l in open(D / "dets.jsonl")]; N = len(rows)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def players(d):
    # a standing person is about 0.25-0.6 as wide as tall: slivers (<0.18) are halves of a person, wide boxes (>0.8) are flags, bags and cones
    return [x[:4] for x in d if x[4] >= 0.5 and x[3] - x[1] >= MINH and Y0 + 5 < x[3] <= Y1 and x[0] > 3 and x[2] < W - 3
            and 0.18 <= (x[2] - x[0]) / (x[3] - x[1]) <= 0.8]


best = None
for r in range(N):
    ps = players(rows[r]["dets"])
    mx = max([iou(a, b) for i, a in enumerate(ps) for b in ps[i + 1:]] or [0])
    key = (abs(len(ps) - EXPECT) if EXPECT else -len(ps), mx)          # EXPECT = number of players on the pitch if known (a single app input), else the frame with the most
    if best is None or key < best[0]:
        best = (key, r, ps)
SEED, seeds = best[1], best[2]
seeds = sorted(seeds, key=lambda b: b[0])
print(f"seed row {SEED}: {len(seeds)} players (max overlap {best[0][1]:.2f})", flush=True)
FR = D / "sam2_frames"; FR.mkdir(exist_ok=True); t0 = time.time()
cap = cv2.VideoCapture(meta["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
frames = []
for i in range(N):
    ok, im = cap.read(); c = im[Y0:Y1]; frames.append(c); cv2.imwrite(str(FR / f"{i:05d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    for _ in range(meta["stride"] - 1):
        cap.grab()
s = frames[SEED].copy()
for k, b in enumerate(seeds):
    cv2.rectangle(s, (int(b[0]), int(b[1] - Y0)), (int(b[2]), int(b[3] - Y0)), (0, 255, 255), 2); cv2.putText(s, str(k + 1), (int(b[0]), int(b[1] - Y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
cv2.imwrite(str(D / "seed.jpg"), s)
dev = device.pick_device(); print("device:", dev, flush=True)
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", device.sam2_checkpoint(), device=dev)
res = {}
with torch.inference_mode(), device.autocast(dev):
    st = pred.init_state(video_path=str(FR), offload_video_to_cpu=True, offload_state_to_cpu=True)
    for oi, b in enumerate(seeds):
        pred.add_new_points_or_box(st, frame_idx=SEED, obj_id=oi, box=np.array([b[0], b[1] - Y0, b[2], b[3] - Y0], np.float32))
    for rev in (False, True):
        for fi, oids, logits in pred.propagate_in_video(st, start_frame_idx=SEED, reverse=rev):
            m = (logits > 0).cpu().numpy()[:, 0]
            for j, oi in enumerate(oids):
                ys, xs = np.where(m[j])
                if len(xs) > 20:
                    res.setdefault(fi, {})[int(oi)] = [float(xs.min()), float(ys.min() + Y0), float(xs.max()), float(ys.max() + Y0)]
            if fi % 50 == 0:
                print(f"  {'back' if rev else 'fwd '} row {fi} ({time.time() - t0:.0f}s)", flush=True)
print(f"tracking done ({time.time() - t0:.0f}s)", flush=True)
json.dump({"seed_row": SEED, "tracks": {str(r): {str(o): b for o, b in d.items()} for r, d in res.items()}}, open(D / "sam2_tracks.json", "w"))


def colour_class(im, b):
    x1, y1, x2, y2 = [int(v) for v in b]; h, w = y2 - y1, x2 - x1
    c = im[max(0, y1 - Y0 + int(.2 * h)):max(1, y1 - Y0 + int(.55 * h)), max(0, x1 + int(.2 * w)):max(1, x2 - int(.2 * w))]
    if c.size == 0:
        return "?"
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3); k = hsv[(hsv[:, 1] >= 90) & (hsv[:, 2] >= 90)]
    if len(k) < 0.15 * len(hsv):
        return "plain"
    hh = float(np.median(k[:, 0]))
    return "orange" if hh < 21 else ("green" if hh < 50 else "other")


rep = [f"{len(seeds)} players seeded on row {SEED}; {N} rows ({N / sfps:.0f} s)"]; cls = {}
for oi in range(len(seeds)):
    seq = [(r, res[r][oi]) for r in sorted(res) if oi in res[r]]
    cl = [colour_class(frames[r], b) for r, b in seq]; cls[oi] = max(set(cl), key=cl.count) if cl else "?"
    main = cls[oi]; flips = sum(1 for a, b in zip(cl, cl[1:]) if a != b and "plain" not in (a, b)); agree = np.mean([c == main for c in cl]) if cl else 0
    jumps = 0
    for (r0, b0), (r1, b1) in zip(seq, seq[1:]):
        if r1 - r0 <= 2:
            h = b0[3] - b0[1]; d = np.hypot((b0[0] + b0[2]) / 2 - (b1[0] + b1[2]) / 2, b0[3] - b1[3])
            jumps += d > 1.2 * h
    rep.append(f"  #{oi + 1}: {len(seq)}/{N} rows, torso colour {main} on {agree * 100:.0f}% of rows, colour flips {flips}, teleports {jumps}")
open(D / "sam2_track_report.txt", "w").write("\n".join(rep)); print("\n".join(rep))
COL = {"orange": (0, 140, 255), "green": (60, 230, 120), "plain": (255, 255, 255), "other": (255, 120, 255), "?": (200, 200, 200)}
S = min(1.0, 1400 / W); vw, _vp = device.video_writer(D / "sam2_track.webm", sfps, (int(W * S), int((Y1 - Y0) * S)))
for r in range(N):
    im = frames[r].copy()
    for oi, b in res.get(r, {}).items():
        c = COL[cls[oi]]; cv2.rectangle(im, (int(b[0]), int(b[1] - Y0)), (int(b[2]), int(b[3] - Y0)), c, 2); cv2.putText(im, f"#{oi + 1}", (int(b[0]), int(b[1] - Y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    cv2.putText(im, f"SAM 2  t={r / sfps:4.1f}s   (box colour = bib colour, labels = player numbers fixed on frame {SEED})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    vw.write(cv2.resize(im, (int(W * S), int((Y1 - Y0) * S))))
vw.release(); print("wrote", D / "sam2_track.webm")
