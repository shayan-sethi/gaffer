#!/usr/bin/env python3
"""
Calibrated top-down map + distances. Needs a homography H (pitch metres -> image pixels), e.g. fitted from pitch lines.
  python pitch_map.py <video> <out_dir> <start_s> <seconds> <n> <H.npy> [pitch_len=105] [pitch_wid=68] [identity=auto]
Players are mapped by their feet. Officials are found by position: someone who spends most of the time on or
outside a touchline / goal line is not a player.
"""
import sys, os, json, math
import numpy as np
import cv2
video, out, st, sec, n, Hf = sys.argv[1:7]
PL = float(sys.argv[7]) if len(sys.argv) > 7 else 105.0
PW = float(sys.argv[8]) if len(sys.argv) > 8 else 68.0
pick = sys.argv[9] if len(sys.argv) > 9 else "auto"
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C

Hinv = np.linalg.inv(np.load(Hf))
meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"])
sfps = meta["fps"] / meta["stride"]
pf, _ = C.track(rows, np.ones((meta["height"], meta["width"]), np.uint8), [], sfps)
ident, un, *_ = C.lock_identities(pf, sfps)


def to_pitch(u, v):
    p = Hinv @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def sm(v, w=7):
    p = np.pad(v, (w // 2, w // 2), mode="edge")
    return np.array([np.median(p[i:i + w]) for i in range(len(v))])


res = []
for k, I in enumerate(ident):
    fr = sorted(I["frames"])
    xy = np.array([to_pitch((I["frames"][f]["box"][0] + I["frames"][f]["box"][2]) / 2, I["frames"][f]["box"][3]) for f in fr])
    t = np.array(fr) / sfps
    X, Y = sm(xy[:, 0]), sm(xy[:, 1])
    dt = np.diff(t); step = np.hypot(np.diff(X), np.diff(Y)); v = step / np.maximum(dt, 1e-6)
    ok = (dt <= 2.0 / sfps + 1e-6) & (v <= 9.0)
    near_line = np.mean((Y < 1.5) | (Y > PW - 1.5) | (X < 1.5) | (X > PL - 1.5) | (Y < 0) | (Y > PW))
    res.append(dict(k=k, I=I, t=t, X=X, Y=Y, dist=float(step[ok].sum()), v=v[ok], near=float(near_line)))
for r in res:
    r["official"] = r["near"] >= 0.7
print("id   team  distance  top speed(1s)  time on/outside a line   verdict")
for r in sorted(res, key=lambda r: -r["near"]):
    vv = r["v"]; top = float(np.percentile(vv, 98)) * 3.6 if len(vv) else 0
    print(f"{r['k']:3d}   {r['I']['team']:>2}   {r['dist']:6.1f} m   {top:5.1f} km/h        {100*r['near']:3.0f}%                 {'OFFICIAL (on the line)' if r['official'] else ''}")
players = [r for r in res if not r["official"]]
k_sel = (max(players, key=lambda r: r["dist"])["k"] if pick == "auto" else int(pick))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 8.4))
ax.add_patch(Rectangle((0, 0), PL, PW, fill=True, color="#2e6b3a", zorder=0))
lw = dict(color="white", lw=1.2, zorder=1)
ax.plot([0, PL, PL, 0, 0], [0, 0, PW, PW, 0], **lw); ax.plot([PL / 2] * 2, [0, PW], **lw)
ax.add_patch(Circle((PL / 2, PW / 2), 9.15, fill=False, **lw))
for x0, x1 in ((0, 16.5), (PL, PL - 16.5)):
    ax.add_patch(Rectangle((min(x0, x1), (PW - 40.32) / 2), 16.5, 40.32, fill=False, **lw))
    ax.add_patch(Rectangle((min(x0, x0 + (5.5 if x0 == 0 else -5.5)), (PW - 18.32) / 2), 5.5, 18.32, fill=False, **lw))
for r in res:
    if r["official"]:
        ax.plot(r["X"], r["Y"], color="#ff5252", lw=1.5, zorder=2); continue
    ax.plot(r["X"], r["Y"], color="white" if r["k"] != k_sel else "none", alpha=0.35, lw=1, zorder=2)
s = [r for r in res if r["k"] == k_sel][0]
sc = ax.scatter(s["X"], s["Y"], c=s["t"], cmap="plasma", s=18, zorder=4)
ax.scatter([s["X"][0]], [s["Y"][0]], s=100, color="white", zorder=5); ax.annotate("start", (s["X"][0], s["Y"][0]), color="white", xytext=(6, 6), textcoords="offset points", zorder=6)
ax.set_xlim(-6, PL + 6); ax.set_ylim(-6, PW + 6); ax.set_aspect("equal"); ax.set_xlabel("metres along the pitch"); ax.set_ylabel("metres across (0 = nearest touchline)")
plt.colorbar(sc, ax=ax, label="seconds into the clip", shrink=0.6)
ax.set_title(f"Player {s['k']}: path on the pitch, {s['dist']:.0f} m in {s['t'][-1]-s['t'][0]:.0f} s   |   faint = other players, red = on the line (officials)", fontsize=11)
fig.text(0.5, 0.01, "Calibrated from the pitch lines by eye (halfway line, centre circle, near touchline and corners); error probably 1-3 m, larger at the far side.", ha="center", fontsize=8, color="#99a")
fig.savefig(out + f"/pitch_map_{k_sel}.png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote", out + f"/pitch_map_{k_sel}.png")
