#!/usr/bin/env python3
"""
Player signatures that do NOT depend on the shirt: grass-free colour of the head/hair, the sleeves/arms, the shorts-trousers and the socks-shoes (Lab),
measured from the video for every locked identity. Used to (1) CHECK identities: does each identity stay one person over time? and
(2) later, to link tracklets across crossings.  Nothing here is set per clip.
  python player_signature.py <out_dir> [window_s=3]  ->  <out_dir>/signatures.json, signature_report.txt, signature_timeline.png
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]); WIN_S = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
meta = json.load(open(D / "meta.json")); ids = json.load(open(D / "identities.json"))
sfps = meta["fps"] / meta["stride"]; WIN = int(WIN_S * sfps)
# parts: (name, y0, y1, [(x0, x1), ...]) as fractions of the box
PARTS = [("head", .00, .15, [(.30, .70)]), ("arms", .20, .50, [(.00, .22), (.78, 1.0)]), ("legs", .52, .76, [(.25, .75)]), ("feet", .82, 1.0, [(.15, .85)])]
W_PART = np.array([1.0, 1.0, 1.3, 0.8])                  # legs are the most telling and the most visible


def lab_of(px):
    p = np.uint8([[np.median(px, 0)]]); L, a, b = cv2.cvtColor(p, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, 1.5 * (a - 128), 1.5 * (b - 128)])


def signature(im, b):
    x1, y1, x2, y2 = [int(v) for v in b[:4]]; h, w = y2 - y1, x2 - x1
    if h < 55 or w < 12:
        return None
    out = []
    for name, ya, yb, xs in PARTS:
        px = []
        for xa, xb in xs:
            c = im[max(0, y1 + int(ya * h)):y1 + int(yb * h), max(0, x1 + int(xa * w)):x1 + int(xb * w)]
            if c.size:
                hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                px.append(c.reshape(-1, 3)[~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 95) & (hsv[:, 1] > 35))])      # drop grass
        px = np.concatenate(px) if px else np.zeros((0, 3))
        out.append(lab_of(px) if len(px) >= 5 else np.full(3, np.nan))
    return np.array(out)                                   # 4 parts x 3


rows_needed = {}
for l, I in ids.items():
    for k, b in I["rows"].items():
        if not b[4] and b[3] - b[1] >= 55:
            rows_needed.setdefault(int(k), []).append((l, b))
cap = cv2.VideoCapture(meta["video"]); sig = {l: {} for l in ids}
last = max(rows_needed); pos = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"]); fr = meta["start_frame"]
for r in range(last + 1):
    ok, im = cap.read()
    if not ok:
        break
    for _ in range(meta["stride"] - 1):
        cap.grab()
    for l, b in rows_needed.get(r, []):
        s = signature(im, b)
        if s is not None:
            sig[l][r] = s
print("signatures measured:", {ids[l]["label"]: len(v) for l, v in sig.items()}, flush=True)


def dist(a, b):                                           # weighted distance over the parts both signatures have
    d = np.linalg.norm(a - b, axis=1); ok = ~np.isnan(d)
    return float(np.sum(d[ok] * W_PART[ok]) / np.sum(W_PART[ok])) if ok.any() else np.nan


def med(S):
    return np.nanmedian(np.array(S), axis=0)


lab = {l: I["label"] for l, I in ids.items()}; team = {l: I["team"] for l, I in ids.items()}
glob = {l: med(list(v.values())) for l, v in sig.items() if v}
n_rows = max(max(map(int, I["rows"])) for I in ids.values()) + 1
nwin = int(np.ceil(n_rows / WIN)); report = []; M = {}
for t in "WB":
    mates = [l for l in glob if team[l] == t]
    between = [dist(glob[a], glob[b]) for i, a in enumerate(mates) for b in mates[i + 1:]]
    report.append(f"TEAM {t}: median distance between two teammates' signatures = {np.nanmedian(between):.1f} (smallest {np.nanmin(between):.1f})")
    for l in mates:
        row = []
        for w in range(nwin):
            S = [s for r, s in sig[l].items() if w * WIN <= r < (w + 1) * WIN]
            if len(S) < 5:
                row.append(-1); continue
            m = med(S); d = {o: dist(m, glob[o]) for o in mates}
            best = min(d, key=lambda o: (np.inf if np.isnan(d[o]) else d[o]))
            row.append(mates.index(best) if (best != l and d[l] - d[best] > 0.15 * np.nanmedian(between)) else mates.index(l))
        M[l] = (mates, row)
        bad = [(w, mates[row[w]]) for w in range(nwin) if row[w] >= 0 and mates[row[w]] != l]
        report.append(f"  {lab[l]}: {sum(1 for x in row if x >= 0)} windows, {len(bad)} look like another player" +
                      ("" if not bad else " -> " + ", ".join(f"{w * WIN_S:.0f}-{(w + 1) * WIN_S:.0f}s looks like {lab[o]}" for w, o in bad)))
open(D / "signature_report.txt", "w").write("\n".join(report)); print("\n".join(report))
json.dump({lab[l]: np.nan_to_num(v, nan=-1).tolist() for l, v in glob.items()}, open(D / "signatures.json", "w"))
fig, axs = plt.subplots(2, 1, figsize=(12, 6.5))
cmap = plt.get_cmap("tab10")
for ax, t in zip(axs, "WB"):
    mates = [l for l in glob if team[l] == t]; img = np.ones((len(mates), nwin, 3))
    for i, l in enumerate(mates):
        for w, v in enumerate(M[l][1]):
            img[i, w] = (0.85, 0.85, 0.85) if v < 0 else cmap(v)[:3]
    ax.imshow(img, aspect="auto", extent=[0, nwin * WIN_S, len(mates), 0]); ax.set_yticks(np.arange(len(mates)) + .5); ax.set_yticklabels([lab[l] for l in mates])
    ax.set_title(f"{'pink bibs' if t == 'W' else 'no bibs'}: colour = whose signature each 3 s window matches best (one steady colour per row = one person; grey = too few samples)", fontsize=9)
axs[1].set_xlabel("seconds into the clip"); fig.tight_layout(); fig.savefig(D / "signature_timeline.png", dpi=100)
print("wrote", D / "signature_timeline.png")
