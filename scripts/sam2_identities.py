#!/usr/bin/env python3
"""
SAM 2 as the identity backbone.  Seed every player with his box on ONE frame where all of them are visible and apart (found automatically from a first
pass of the locker, or given), follow each mask forward AND backward through the window, then give each SAM 2 object his team from the bib colour.
  python sam2_identities.py <out_dir> <n_rows> [seed_row]     -> <out_dir>/identities_sam2.json (same format as identities.json), sam2_report.txt
Team = median bib share of the object over the window; split at the largest gap between the 12 medians (no constants).
A bib reading that flips inside one object's track is reported as a possible drift onto another player.
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, cv2, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(next((ROOT / ".local_data/ext/sam2_src").glob("sam2-*"))))
from sam2.build_sam import build_sam2_video_predictor

D = Path(sys.argv[1]); N = int(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else None
meta = json.load(open(D / "meta.json")); seeds = json.load(open(D / "identities.json"))
Y0 = int(meta.get("band_y0", 0)); Y1 = meta["height"]
dets = [json.loads(l)["dets"] for l in open(D / "dets.jsonl")][:N]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


keys = sorted(seeds)
if SEED is None:                                       # a row where everyone is seen, big enough, and nobody overlaps anybody
    best = None
    for r in range(N):
        bs = [seeds[k]["rows"].get(str(r)) for k in keys]
        if all(b and not b[4] and b[3] - b[1] >= 60 for b in bs):
            mx = max(iou(a, b) for i, a in enumerate(bs) for b in bs[i + 1:])
            if best is None or mx < best[0]:
                best = (mx, r)
    SEED = best[1]
print("seed row", SEED, flush=True)
OUT = D / "sam2_run"; FR = OUT / "frames"; FR.mkdir(parents=True, exist_ok=True)
t0 = time.time()
if len(list(FR.glob("*.jpg"))) < N:
    cap = cv2.VideoCapture(meta["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
    for i in range(N):
        ok, im = cap.read()
        if not ok:
            N = i; break
        cv2.imwrite(str(FR / f"{i:05d}.jpg"), im[Y0:Y1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        for _ in range(meta["stride"] - 1):
            cap.grab()
print(f"{N} frames ready ({time.time() - t0:.0f}s)", flush=True)
dev = "cuda"
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", str(ROOT / ".local_data/ext/sam2_ckpt/sam2.1_hiera_small.pt"), device=dev)
res = {}                                               # row -> {obj: [x1,y1,x2,y2]}
with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
    st = pred.init_state(video_path=str(FR), offload_video_to_cpu=True, offload_state_to_cpu=True)
    for oi, k in enumerate(keys):
        b = seeds[k]["rows"][str(SEED)]
        pred.add_new_points_or_box(st, frame_idx=SEED, obj_id=oi, box=np.array([b[0], b[1] - Y0, b[2], b[3] - Y0], np.float32))
    for rev in (False, True):
        for fi, oids, logits in pred.propagate_in_video(st, start_frame_idx=SEED, reverse=rev):
            m = (logits > 0).cpu().numpy()[:, 0]
            for j, oi in enumerate(oids):
                ys, xs = np.where(m[j])
                if len(xs) > 20:
                    res.setdefault(fi, {})[int(oi)] = [float(xs.min()), float(ys.min() + Y0), float(xs.max()), float(ys.max() + Y0)]
            if fi % 40 == 0:
                print(f"  {'back' if rev else 'fwd '} row {fi} ({time.time() - t0:.0f}s, GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
print(f"tracking done ({time.time() - t0:.0f}s)", flush=True)

# bib share per object per row (from the detection that overlaps the mask box most)
share = {oi: {} for oi in range(len(keys))}
for r, d in res.items():
    for oi, b in d.items():
        best = max(dets[r], key=lambda x: iou(b, x[:4]), default=None)
        if best is not None and iou(b, best[:4]) > 0.4 and len(best) > 16 and best[16] >= 0:
            share[oi][r] = best[16]
med = {oi: float(np.median(list(v.values()))) if v else 0.0 for oi, v in share.items()}
order = sorted(med, key=lambda o: -med[o]); gaps = [med[order[i]] - med[order[i + 1]] for i in range(len(order) - 1)]
cut = int(np.argmax(gaps)) + 1                         # the largest gap between the sorted medians separates "wears a bib" from "does not"
thr = (med[order[cut - 1]] + med[order[cut]]) / 2
team = {oi: ("W" if med[oi] >= thr else "B") for oi in med}
rep = [f"seed row {SEED}; bib/no-bib split at median bib share {thr:.2f} (largest gap {max(gaps):.2f}) -> {sum(t == 'W' for t in team.values())} pink, {sum(t == 'B' for t in team.values())} no bib"]
# labels: by team, left to right at the seed row
lab = {}
for t, pre in (("W", "P"), ("B", "B")):
    mem = sorted([oi for oi in team if team[oi] == t], key=lambda o: (res.get(SEED, {}).get(o, [0])[0]))
    for n, oi in enumerate(mem, 1):
        lab[oi] = f"{pre}{n}"
out = {}
for oi in sorted(lab, key=lambda o: lab[o]):
    rows = {str(r): res[r][oi] + [0] for r in sorted(res) if oi in res[r]}
    out[str(oi)] = dict(label=lab[oi], team=team[oi], role="P", rows=rows)
    sh = list(share[oi].values()); on = np.array(sh) >= thr
    flips = int(np.sum(on[1:] != on[:-1])) if len(on) > 1 else 0
    pure = float(np.mean(on == (team[oi] == "W"))) if len(on) else float("nan")
    rep.append(f"  {lab[oi]}: tracked on {len(rows)}/{N} rows, median bib share {med[oi]:.2f}, bib reading agrees with his team on {pure * 100:.0f}% of rows, flips {flips}")
json.dump(out, open(D / "identities_sam2.json", "w")); open(D / "sam2_report.txt", "w").write("\n".join(rep)); print("\n".join(rep))
