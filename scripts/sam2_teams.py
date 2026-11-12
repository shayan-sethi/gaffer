#!/usr/bin/env python3
"""
Teams from bib colour, on top of SAM 2 identities (the tracks are NOT touched).
For every identity: the hue of the strong-colour pixels (the bib) on his torso over all rows he is tracked; the two teams are the two groups of
players separated by the largest gap between the players' median hues. Team names come from the hue (red/orange, yellow/green, ...).
  python sam2_teams.py <out_dir>     reads identities.json + sam2_frames/ ; writes identities.json with team/label (backup: identities_noteams.json)
"""
import sys, json, shutil
from pathlib import Path
import numpy as np, cv2

D = Path(sys.argv[1]); meta = json.load(open(D / "meta.json")); Y0 = int(meta["band_y0"])
src = D / "identities_noteams.json"
if not src.exists():
    shutil.copy(D / "identities.json", src)
ids = json.load(open(src)); hue = {}; share = {}
for k, I in ids.items():
    hs, ok = [], 0
    for r, b in I["rows"].items():
        im = cv2.imread(str(D / "sam2_frames" / f"{int(r):05d}.jpg"))
        if im is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in b[:4]]; h, w = y2 - y1, x2 - x1
        c = im[max(0, y1 - Y0 + int(.2 * h)):max(1, y1 - Y0 + int(.55 * h)), max(0, x1 + int(.2 * w)):max(1, x2 - int(.2 * w))]
        if c.size == 0:
            continue
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3); m = (hsv[:, 1] >= 110) & (hsv[:, 2] >= 110)
        if m.mean() >= 0.15:
            hs.append(float(np.median(hsv[m, 0])))
    hue[k] = float(np.median(hs)) if hs else float("nan"); share[k] = len(hs)
order = sorted([k for k in hue if hue[k] == hue[k]], key=lambda k: hue[k])
gaps = [hue[order[i + 1]] - hue[order[i]] for i in range(len(order) - 1)]
cut = int(np.argmax(gaps)) + 1; lowset = set(order[:cut]); thr = (hue[order[cut - 1]] + hue[order[cut]]) / 2
def name(hh):
    return "R" if hh < 22 else ("Y" if hh < 50 else "T")
tn = {0: name(np.median([hue[k] for k in order[:cut]])), 1: name(np.median([hue[k] for k in order[cut:]]))}
if tn[0] == tn[1]:
    tn = {0: "R", 1: "Y"}
rep = [f"bib hue split at {thr:.0f} (largest gap {max(gaps):.0f}): {cut} players {tn[0]} (low hue, median {np.median([hue[k] for k in order[:cut]]):.0f}) and {len(order) - cut} players {tn[1]} (median {np.median([hue[k] for k in order[cut:]]):.0f})"]
first = {k: next(iter(sorted(map(int, I["rows"]))), 0) for k, I in ids.items()}
out = {}; cnt = {tn[0]: 0, tn[1]: 0}
for k in sorted(ids, key=lambda k: np.mean([(b[0] + b[2]) / 2 for b in ids[k]["rows"].values()])):
    t = tn[0] if k in lowset else tn[1]; cnt[t] += 1; I = dict(ids[k]); I["team"] = t; I["label"] = f"{t}{cnt[t]}"; out[k] = I
    rep.append(f"  {I['label']} (was {ids[k]['label']}): median bib hue {hue[k]:.0f} on {share[k]} rows, {len(I['rows'])} tracked")
json.dump(out, open(D / "identities.json", "w")); open(D / "teams_report.txt", "w").write("\n".join(rep)); print("\n".join(rep))
