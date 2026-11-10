#!/usr/bin/env python3
"""
Rough top-down (2D) map and distance for the players of a fixed-camera clip, WITHOUT pitch calibration.

People are the ruler: for a camera looking at a ground plane, a person's box height is proportional to the
distance between their feet and the horizon:  h_px = a * (foot_y - horizon_y)  with  a = 1.75 / camera_height.
A robust fit over all detections gives the horizon and the camera height. With an ASSUMED horizontal field of view
(default 90 degrees) every foot point is then turned into ground metres by intersecting its ray with the ground.
Approximate: wrong FOV or lens distortion scales the depth axis. A real 4-corner calibration replaces all of this.

python topdown_map.py <video> <out_dir> <start_s> <seconds> <n> [identity=auto] [fov_deg=90]
"""
import sys, os, json, math
import numpy as np
import cv2

video, out, st, sec, n = sys.argv[1:6]
pick = sys.argv[6] if len(sys.argv) > 6 else "auto"
FOV = float(sys.argv[7]) if len(sys.argv) > 7 else 90.0
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C

meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"])
sfps = meta["fps"] / meta["stride"]
W, Hh = meta["width"], meta["height"]
pf, _ = C.track(rows, np.ones((Hh, W), np.uint8), [], sfps)
ident, un, *_ = C.lock_identities(pf, sfps)
if not os.environ.get("KEEP_ALL"):
    ident = [I for I in ident if I["role"] != "OFFICIAL"]      # KEEP_ALL=1 keeps flagged "officials" (the flags are unreliable)

# ---- camera from the people
fy, hh = [], []
for I in ident:
    for o in I["frames"].values():
        if not o.get("interp") and 40 <= C.height(o):
            fy.append(o["box"][3]); hh.append(C.height(o))
fy, hh = np.array(fy), np.array(hh)
a, b = np.polyfit(fy, hh, 1)
for _ in range(5):                                  # trim outliers and refit
    res = np.abs(hh - (a * fy + b)); keep = res < 2.0 * np.median(res) + 1e-6
    a, b = np.polyfit(fy[keep], hh[keep], 1)
horizon = -b / a
cam_h = 1.75 / a
f = (W / 2) / math.tan(math.radians(FOV / 2))
cx, cy = W / 2, Hh / 2
tilt = math.atan2(cy - horizon, f)
CYL = os.environ.get('PROJ') == 'cyl'
if CYL:
    f = W / math.radians(FOV)
    tilt = math.atan2(cy - horizon, f)                  # camera pitched down by this angle
print(f"horizon at y={horizon:.0f}px, camera height ~{cam_h:.1f} m, tilt ~{math.degrees(tilt):.0f} deg (assumed FOV {FOV:.0f})")


CYL = os.environ.get("PROJ") == "cyl"      # stitched panoramas are roughly cylindrical, not pinhole
if CYL:
    f = W / math.radians(FOV)              # pixels per radian of azimuth


def ground(u, v):
    """foot pixel -> (X lateral, Z forward) in metres on the ground, origin under the camera"""
    if CYL:
        az = (u - cx) / f
        d = np.array([math.sin(az), (v - cy) / f, math.cos(az)])
    else:
        d = np.array([(u - cx) / f, (v - cy) / f, 1.0])            # camera ray (x right, y down, z forward)
    c, s = math.cos(tilt), math.sin(tilt)
    # rotate by the downward tilt: world y is down, world z is forward
    y_w, z_w = d[1] * c + d[2] * s, -d[1] * s + d[2] * c
    if y_w <= 1e-6:
        return None
    t = cam_h / y_w
    return d[0] * t, z_w * t


def track_metres(I):
    fr = sorted(I["frames"])
    pts = []
    for f_ in fr:
        o = I["frames"][f_]
        g = ground((o["box"][0] + o["box"][2]) / 2, o["box"][3])
        pts.append((f_, g))
    ok = [(f_, g) for f_, g in pts if g is not None]
    t = np.array([p[0] for p in ok]) / sfps
    X = np.array([p[1][0] for p in ok]); Z = np.array([p[1][1] for p in ok])
    def sm(v, w=7):
        p = np.pad(v, (w // 2, w // 2), mode="edge"); return np.array([np.median(p[i:i + w]) for i in range(len(v))])
    return t, sm(X), sm(Z)


def distance(t, X, Z):
    dt = np.diff(t); step = np.hypot(np.diff(X), np.diff(Z)); v = step / np.maximum(dt, 1e-6)
    ok = (dt <= 2.0 / sfps + 1e-6) & (v <= 9.0)
    return float(step[ok].sum()), v[ok]


res = []
for k, I in enumerate(ident):
    t, X, Z = track_metres(I)
    d, v = distance(t, X, Z)
    res.append((k, I, t, X, Z, d))
if pick == "auto":
    outfield = [r for r in res if r[1]["role"] == "P"]
    k_sel = max(outfield, key=lambda r: r[5])[0]
else:
    k_sel = int(pick)
for k, I, t, X, Z, d in res:
    print(f"  {I['team']}{k}: {d:6.1f} m  ({len(I['frames'])/len(rows)*100:3.0f}% of the minute visible)" + ("   <-- shown" if k == k_sel else ""))

# ---- the 2D picture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(9, 8))
allX = np.concatenate([r[3] for r in res]); allZ = np.concatenate([r[4] for r in res])
print(f"extent of all positions: across {np.percentile(allX,1):.0f}..{np.percentile(allX,99):.0f} m, away {np.percentile(allZ,1):.0f}..{np.percentile(allZ,99):.0f} m")
for k, I, t, X, Z, d in res:
    col = "#dddddd" if I["team"] == "W" else "#ff9f43"
    ax.plot(X, Z, color=col, alpha=0.28 if k != k_sel else 0.0, lw=1)
_, I, t, X, Z, d = [r for r in res if r[0] == k_sel][0]
sc = ax.scatter(X, Z, c=t, cmap="viridis", s=14, zorder=3)
ax.plot(X, Z, color="white", lw=0.8, alpha=0.6, zorder=2)
ax.scatter([X[0]], [Z[0]], marker="o", s=90, color="white", zorder=4); ax.annotate("start", (X[0], Z[0]), color="white", xytext=(6, 6), textcoords="offset points")
ax.scatter([0], [0], marker="^", s=120, color="#4fc3f7", zorder=5); ax.annotate("camera", (0, 0), color="#4fc3f7", xytext=(6, -14), textcoords="offset points")
ax.set_aspect("equal"); ax.set_xlabel("metres across (left/right of the camera)"); ax.set_ylabel("metres away from the camera")
plt.colorbar(sc, ax=ax, label="seconds into the clip", shrink=0.7)
ax.set_title(f"Player {I['team']}{k_sel}: path seen from above (approximate)\ndistance {d:.0f} m in {t[-1]-t[0]:.0f} s   |   faint lines = the other players", fontsize=11)
ax.grid(alpha=0.2)
fig.text(0.5, 0.005, f"Approximate: no pitch calibration. Camera height ~{cam_h:.1f} m and tilt ~{math.degrees(tilt):.0f} deg estimated from player heights; assumes a {FOV:.0f} degree field of view.",
         ha="center", fontsize=8, color="#889")
fig.savefig(out + f"/topdown_{k_sel}.png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote", out + f"/topdown_{k_sel}.png")
# sensitivity of the distance to the field-of-view guess is checked by re-running with a different fov_deg
