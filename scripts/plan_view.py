#!/usr/bin/env python3
"""
Approximate top-down (plan) map of average positions for a fixed camera WITHOUT a pitch calibration.
Ground plane from the players themselves (box height vs foot row -> camera height + horizon), an ASSUMED lens field of view, then rotated so the line
between the two goals is horizontal. The along-pitch order and the front-to-back order are reliable; absolute metres are not (+-30%).
  python plan_view.py <out_dir> <fov_deg> <Lgoal_u,v,u,v> <Rgoal_u,v,u,v> [outfile]      goals = two image points each (near post base, far post base)
"""
import sys, json, math, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

D = Path(sys.argv[1]); FOV = float(sys.argv[2])
LG = [float(v) for v in sys.argv[3].split(",")]; RG = [float(v) for v in sys.argv[4].split(",")]
OUTF = Path(sys.argv[5]) if len(sys.argv) > 5 else D / "report" / "plan_view.png"
W, H = 1920, 1080
meta = json.load(open(D / "meta.json")); ids = json.load(open(D / "identities.json"))
ev = json.load(open(D / "pass_events.json")); spells = json.load(open(D / "pass_spells.json"))
trk = {}
for r in csv.DictReader(open(D / "ball_track.csv")):
    if r["x"] != "" and r["source"] in ("det", "interp"):
        trk[int(round(float(r["frame"])))] = (float(r["x"]), float(r["y"]))
fy, hh = [], []
for I in ids.values():
    for b in I["rows"].values():
        if not b[4] and b[3] - b[1] >= 40:
            fy.append(b[3]); hh.append(b[3] - b[1])
fy, hh = np.array(fy), np.array(hh); a, b0 = np.polyfit(fy, hh, 1)
for _ in range(5):
    r = np.abs(hh - (a * fy + b0)); k = r < 2 * np.median(r) + 1e-6; a, b0 = np.polyfit(fy[k], hh[k], 1)
hor, ch = -b0 / a, 1.75 / a
f = (W / 2) / math.tan(math.radians(FOV / 2)); cx, cy = W / 2, H / 2; tilt = math.atan2(cy - hor, f)


def ground(u, v):
    d = np.array([(u - cx) / f, (v - cy) / f, 1.0]); c, s = math.cos(tilt), math.sin(tilt)
    yw, zw = d[1] * c + d[2] * s, -d[1] * s + d[2] * c
    t = ch / yw
    return np.array([d[0] * t, zw * t])


gl = (ground(*LG[:2]) + ground(*LG[2:])) / 2; gr = (ground(*RG[:2]) + ground(*RG[2:])) / 2
L = float(np.linalg.norm(gr - gl)); u = (gr - gl) / L; n = np.array([-u[1], u[0]])
mid = (gl + gr) / 2


def to_pitch(p):                       # x along the pitch from the left goal, y across (positive = away from the camera)
    q = p - gl
    return np.array([float(np.dot(q, u)), float(np.dot(q, n))])
if to_pitch(ground(960, 600))[1] > 0:  # make "away from the camera" positive y
    pass
sgn = 1.0 if to_pitch(ground(960, 440))[1] > to_pitch(ground(960, 700))[1] else -1.0


def P(u_, v_):
    q = to_pitch(ground(u_, v_)); return np.array([q[0], sgn * q[1]])


pl = {}
for l, I in ids.items():
    pts = []
    for fr, bx in I["rows"].items():
        pts.append(P((bx[0] + bx[2]) / 2, bx[3]))
    pl[I["label"]] = (np.array(pts), I["team"])
mean = {l: v[0].mean(0) for l, v in pl.items()}
cov = {l: np.cov(v[0].T) for l, v in pl.items()}
far_y = np.mean([P(x_, 440)[1] for x_ in (600, 960, 1300)])
near_y = min(v[0][:, 1].min() for v in pl.values())
TCOL = {"W": "#ff5fa8", "B": "#4aa8ff"}
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12.5, 6.8))
ax.add_patch(plt.Rectangle((0, near_y - 1.5), L, far_y - (near_y - 1.5), color="#2e6b3a", zorder=0))
ax.plot([0, L, L, 0, 0], [near_y - 1.5, near_y - 1.5, far_y, far_y, near_y - 1.5], color="white", lw=1.2, ls="--", alpha=0.5)
for gx in (0, L):
    ax.plot([gx, gx], [-1.8, 1.8], color="white", lw=5, solid_capstyle="butt", zorder=2)
ax.text(0, far_y + 0.4, "left goal", ha="center", fontsize=9, color="#cde"); ax.text(L, far_y + 0.4, "right goal", ha="center", fontsize=9, color="#cde")
lab2t = {l: v[1] for l, v in pl.items()}; cnt = {}
for e in ev:
    if e["kind"] == "pass":
        cnt[(e["frm"], e["to"])] = cnt.get((e["frm"], e["to"]), 0) + 1
for (a_, b_), c in cnt.items():
    ax.add_patch(FancyArrowPatch(tuple(mean[a_]), tuple(mean[b_]), arrowstyle="-|>", mutation_scale=14 + 5 * c, lw=1.2 + 1.8 * c, color="#ffd54a", alpha=0.8, connectionstyle="arc3,rad=0.25", shrinkA=11, shrinkB=11, zorder=3))
for l, (pts, t) in pl.items():
    ev_, vec = np.linalg.eigh(cov[l]); ang = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(tuple(mean[l]), 2 * math.sqrt(max(ev_[1], 1e-6)), 2 * math.sqrt(max(ev_[0], 1e-6)), angle=ang, color=TCOL[t], alpha=0.25, zorder=1))
    ax.scatter(*mean[l], s=430, color=TCOL[t], edgecolor="white", zorder=4); ax.text(mean[l][0], mean[l][1], l, ha="center", va="center", fontsize=8.5, color="black", weight="bold", zorder=5)
bp = np.array([P(*v) for v in trk.values()])
if len(bp):
    ax.plot(bp[:, 0], bp[:, 1], color="white", lw=0.7, alpha=0.45, zorder=2); ax.scatter(*bp.mean(0), s=90, color="white", marker="o", edgecolor="black", zorder=6); ax.text(bp.mean(0)[0] + 0.5, bp.mean(0)[1] + 0.5, "ball (average)", fontsize=8, color="white")
ax.set_aspect("equal"); ax.set_xlim(-2.5, L + 2.5); ax.set_ylim(near_y - 2.5, far_y + 2)
ax.set_xlabel("metres along the pitch (approximate)"); ax.set_ylabel("metres across, away from the camera (approximate)")
ax.set_title(f"Average positions, seen from above (pink = pink bibs, blue = no bibs; ellipse = where each player moved; yellow arrows = passes)\n"
             f"APPROXIMATE: scale from player heights and an assumed lens ({FOV:.0f} deg); goal-to-goal about {L:.0f} m; the near edge of the pitch is not visible, so it is drawn dashed", fontsize=9.5)
OUTF.parent.mkdir(exist_ok=True, parents=True); fig.savefig(OUTF, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
print("camera height", round(ch, 2), "m | goal-to-goal", round(L, 1), "m | wrote", OUTF)
for l in sorted(mean, key=lambda l: (lab := lab2t[l], l)): print(f"  {l}: ({mean[l][0]:5.1f} m along, {mean[l][1]:5.1f} m across)")
