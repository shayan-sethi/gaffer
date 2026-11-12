#!/usr/bin/env python3
"""
Per-player metrics for a calibrated clip, from identities.json (written by clip_passes.py) + the pitch homography.
  python player_metrics.py <out_dir> <fps_sampled> <H.npy> [pitch_len=105] [pitch_wid=68]
Writes <out_dir>/player_metrics.csv, player_metrics.json and player_grid.png.
Distances/speeds are in real metres (calibration read from the pitch lines by eye: roughly 1-3 m of error near the
camera, more on the far side, so far-side speeds are the noisiest).
"""
import sys, json, csv, math
import numpy as np
OUT, SFPS, HF = sys.argv[1], float(sys.argv[2]), sys.argv[3]
PL = float(sys.argv[4]) if len(sys.argv) > 4 else 105.0
PW = float(sys.argv[5]) if len(sys.argv) > 5 else 68.0
OUT = OUT.rstrip("/")
Hinv = np.linalg.inv(np.load(HF))
ids = json.load(open(OUT + "/identities.json"))
passes = json.load(open(OUT + "/passes.json"))
pl = passes["players"]                       # keyed by label: possessions, time_on_ball_s, passes_made, passes_received

import os as _os
NOPITCH = bool(_os.environ.get("NOPITCH"))
LANDMARKS = json.loads(_os.environ.get("LANDMARKS", "[]"))          # [[name, u, v], ...] image points to show on the map when the pitch lines are unknown
def landmark_xy(Hinv):
    out = []
    for name, u, v in LANDMARKS:
        p = Hinv @ np.array([u, v, 1.0]); out.append((name, p[0] / p[2], p[1] / p[2]))
    return out
ZONES = [("standing", 0, 0.5), ("walking", 0.5, 1.94), ("jogging", 1.94, 3.89), ("running", 3.89, 5.83), ("sprinting", 5.83, 99)]
MAXV = 9.0


def sm(v, w=7):
    p = np.pad(v, (w // 2, w // 2), mode="edge"); return np.array([np.median(p[i:i + w]) for i in range(len(v))])


def runs(mask, nmin):
    r = c = 0
    for v in list(mask) + [False]:
        if v: c += 1
        else: r += c >= nmin; c = 0
    return int(r)


res = []
for k, I in ids.items():
    if I["role"] == "REF":
        continue
    fr = sorted(int(f) for f in I["rows"])
    est = sum(1 for f in fr if I["rows"][str(f)][4])
    xy = []
    for f in fr:
        b = I["rows"][str(f)]
        p = Hinv @ np.array([(b[0] + b[2]) / 2, b[3], 1.0]); xy.append((p[0] / p[2], p[1] / p[2]))
    xy = np.array(xy); t = np.array(fr) / SFPS
    X, Y = sm(xy[:, 0]), sm(xy[:, 1])
    dt = np.diff(t); step = np.hypot(np.diff(X), np.diff(Y)); v = step / np.maximum(dt, 1e-6)
    ok = (dt <= 2.0 / SFPS + 1e-6) & (v <= MAXV)
    w = max(2, int(round(SFPS)))
    inst = np.where(ok, v, np.nan); sp = np.full(len(fr), np.nan)
    for i in range(len(inst)):
        win = inst[max(0, i - w + 1):i + 1]
        if np.isfinite(win).sum() >= 0.6 * w: sp[i + 1] = np.nanmean(win)
    g = np.isfinite(sp)
    zone = {n: (round(100 * float(np.mean((sp[g] >= lo) & (sp[g] < hi)))) if g.any() else 0) for n, lo, hi in ZONES}
    dist = float(step[ok].sum()); seen = len(fr) / SFPS
    lab = I["label"]; pm = pl.get(lab, {})
    res.append(dict(label=lab, team=I["team"], role=I["role"], seen_s=round(seen, 1), estimated_pct=round(100 * est / len(fr)),
                    distance_m=round(dist, 1), avg_speed_kmh=round(float(np.nanmean(sp)) * 3.6, 1) if g.any() else None,
                    top_speed_kmh=round(min(float(np.nanmax(sp)) * 3.6, 36.0), 1) if g.any() else None,
                    sprints=runs(np.nan_to_num(sp, nan=0) >= 5.83, int(0.8 * SFPS)), **{f"{n}_pct": zone[n] for n, _, _ in ZONES},
                    avg_x=round(float(X.mean()), 1), avg_y=round(float(Y.mean()), 1), spread_x=round(float(X.std()), 1), spread_y=round(float(Y.std()), 1),
                    possessions=pm.get("possessions", 0), time_on_ball_s=pm.get("time_on_ball_s", 0.0),
                    passes_made=pm.get("passes_made", 0), passes_received=pm.get("passes_received", 0),
                    jumps=int(np.sum((dt <= 2.0 / SFPS + 1e-6) & (v > MAXV))), _X=X, _Y=Y, _t=t, _sp=sp, _ok=ok))
order = {"W": 0, "B": 1}
res.sort(key=lambda r: (order.get(r["team"], 2), r["role"] != "GK", -r["distance_m"]))
cols = [c for c in res[0] if not c.startswith("_")]
with open(OUT + "/player_metrics.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
    for r in res: wr.writerow({c: r[c] for c in cols})
json.dump([{c: r[c] for c in cols} for r in res], open(OUT + "/player_metrics.json", "w"), indent=1)
print(f"{'player':8s}{'seen':>6s}{'dist m':>8s}{'avg km/h':>9s}{'top km/h':>9s}{'sprints':>8s}{'jog%':>6s}{'run%':>6s}{'avg pos (x,y)':>16s}{'on ball s':>10s}{'passes':>7s}{'recv':>5s}")
for r in res:
    print(f"{r['label']:8s}{r['seen_s']:6.0f}{r['distance_m']:8.1f}{(r['avg_speed_kmh'] or 0):9.1f}{(r['top_speed_kmh'] or 0):9.1f}{r['sprints']:8d}{r['jogging_pct']:6d}{r['running_pct']+r['sprinting_pct']:6d}   ({r['avg_x']:4.0f},{r['avg_y']:3.0f})   {r['time_on_ball_s']:8.1f}{r['passes_made']:7d}{r['passes_received']:5d}  jumps {r['jumps']}")

# ---- grid of mini pitches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
plt.style.use("dark_background")
LM = landmark_xy(Hinv)
allx = np.concatenate([r['_X'] for r in res] + [np.array([l[1] for l in LM])]); ally = np.concatenate([r['_Y'] for r in res] + [np.array([l[2] for l in LM])])
XMIN, XMAX, YMIN, YMAX = float(allx.min()) - 2, float(allx.max()) + 2, float(ally.min()) - 2, float(ally.max()) + 2
n = len(res); ncol = 5; nrow = math.ceil(n / ncol)
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4.4, nrow * 3.3))
for ax in axs.ravel(): ax.axis("off")
for ax, r in zip(axs.ravel(), res):
    if NOPITCH:
        ax.add_patch(Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN, color="#2e6b3a", zorder=0))
        for nm, lx, ly in LM:
            ax.scatter([lx], [ly], s=18, color="#ffd54a", zorder=2); ax.annotate(nm, (lx, ly), color="#ffd54a", fontsize=6, xytext=(3, 3), textcoords="offset points")
    else:
      ax.add_patch(Rectangle((0, 0), PL, PW, color="#2e6b3a", zorder=0))
      lw = dict(color="white", lw=0.6, zorder=1)
      ax.plot([0, PL, PL, 0, 0], [0, 0, PW, PW, 0], **lw); ax.plot([PL / 2] * 2, [0, PW], **lw)
      ax.add_patch(Circle((PL / 2, PW / 2), 9.15, fill=False, **lw))
      for x0 in (0, PL - 16.5): ax.add_patch(Rectangle((x0, (PW - 40.32) / 2), 16.5, 40.32, fill=False, **lw))
    col = {"W": "#ffffff", "B": "#4aa8ff"}.get(r["team"], "#aaaaaa")
    Xp, Yp = r["_X"].copy(), r["_Y"].copy(); bad = np.where(~r["_ok"])[0]; Xp[bad] = np.nan; Yp[bad] = np.nan      # break the line at jumps
    ax.scatter(Xp, Yp, c=r["_t"], cmap="plasma", s=5, zorder=3)
    ax.plot(Xp, Yp, color=col, lw=0.8, alpha=0.7, zorder=2)
    ax.set_xlim((XMIN - 1, XMAX + 1) if NOPITCH else (-3, PL + 3)); ax.set_ylim((YMIN - 1, YMAX + 1) if NOPITCH else (-3, PW + 3)); ax.set_aspect("equal")
    ax.set_title(f"{r['label']}  {r['distance_m']:.0f} m  top {r['top_speed_kmh']:.0f} km/h", fontsize=10, color=col)
fig.suptitle("Each player's path, seen from above (colour = time)   |   metres and speeds are ESTIMATES (camera model from player heights, no pitch lines)" if NOPITCH else "Each player's path on the pitch (30 s, colour = time)   |   metres and speeds are estimates from a hand-fitted calibration", fontsize=11, y=0.995)
fig.tight_layout(); fig.savefig(OUT + "/player_grid.png", dpi=75, facecolor=fig.get_facecolor())
import pickle
pickle.dump(res, open(OUT + "/player_series.pkl", "wb"))
print("wrote player_metrics.csv / .json / player_grid.png / player_series.pkl")
