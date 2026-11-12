#!/usr/bin/env python3
"""Run the Deep-EIoU tracker (hsiangwei0903/Deep-EIoU, patched to run without compiled extras; source in .local_data/ext/deepeiou/pkg)
on the cached person detections of a clip and compare it with my identity locker.
  python deepeiou_test.py <out_dir> [pitch_mask.npy]     -> <out_dir>/deepeiou_tracks.json
Appearance embedding = grass-free Lab shirt colour + strong-colour shares already stored in dets.jsonl (no model needed)."""
import sys, json, types
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".local_data/ext/deepeiou/pkg"))
from tracker.Deep_EIoU import Deep_EIoU

D = Path(sys.argv[1]); MASK = np.load(sys.argv[2]) if len(sys.argv) > 2 else None
rows = sorted((json.loads(l) for l in open(D / "dets.jsonl")), key=lambda r: r["frame"])
args = types.SimpleNamespace(track_high_thresh=0.4, track_low_thresh=0.1, new_track_thresh=0.45, track_buffer=60, match_thresh=0.8,
                             proximity_thresh=0.5, appearance_thresh=0.25, with_reid=True)
trk = Deep_EIoU(args, frame_rate=30)
out = {}                                                     # row -> [[tid, x1,y1,x2,y2], ...]
for i, r in enumerate(rows):
    ds = []
    for d in r["dets"]:
        if d[4] < 0.3:
            continue
        fx, fy = int((d[0] + d[2]) / 2), int(d[3])
        if MASK is not None and not (0 <= fy < MASK.shape[0] and 0 <= fx < MASK.shape[1] and MASK[fy, fx]):
            continue
        ds.append(d)
    if not ds:
        trk.update(np.zeros((0, 5)), np.zeros((0, 9))); continue
    A = np.array([d[:5] for d in ds], float)
    E = np.array([[d[8] / 60.0, d[9] / 60.0, d[10] / 60.0] + [2 * v for v in d[11:17]] for d in ds], float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    tr = trk.update(A, E)
    out[i] = [[int(t.track_id)] + [float(v) for v in t.tlbr] for t in tr]
json.dump(out, open(D / "deepeiou_tracks.json", "w"))
ids = {}
for i, lst in out.items():
    for t in lst:
        ids.setdefault(t[0], []).append(i)
n = len(rows); lens = sorted((len(v) for v in ids.values()), reverse=True)
print(f"{len(ids)} track ids over {n} rows; longest 12 cover {sum(lens[:12]) / n:.1f} of 12 players per row on average (locker: 10.1); ids under 3 s: {sum(1 for v in lens if v < 30)}")
# team purity against my locker's P/B labels
idn = json.load(open(D / "identities.json")); team = {l: I["team"] for l, I in idn.items()}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
mixed = 0; checked = 0; flips = 0
for tid, rws in ids.items():
    seq = []
    for i in rws:
        box = next(t[1:] for t in out[i] if t[0] == tid)
        best = max(((iou(box, I["rows"][str(i)][:4]), I["team"]) for I in idn.values() if str(i) in I["rows"]), default=(0, None))
        if best[0] > 0.5:
            seq.append(best[1])
    if len(seq) >= 20:
        checked += 1; flips += sum(1 for a, b in zip(seq, seq[1:]) if a != b); mixed += len(set(seq)) > 1
print(f"tracks checked {checked}: {mixed} contain both teams (identity swaps across teams), team flips inside tracks: {flips}")
