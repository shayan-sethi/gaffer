#!/usr/bin/env python3
"""Tune the camera model with the user's marked boundary lines. Camera height comes from the players (height vs foot position);
the field of view and horizon are searched so that, in real metres, the left and right touchlines are PARALLEL and the far wall is
PERPENDICULAR to them. Then writes the homography (ground metres -> image) and reports the pitch width."""
import sys, os, json, math
import numpy as np, cv2
D = ".local_data/tolworth_1080d"
meta = json.load(open(D + "/meta.json")); W, H = meta["width"], meta["height"]
FL, FR, RE, LL = np.load(".local_data/tolworth_user_lines.npy")
ids = json.load(open(D + "/identities.json"))
fy, hh = [], []
for I in ids.values():
    for b in I["rows"].values():
        h = b[3] - b[1]
        if not b[4] and h >= 40: fy.append(b[3]); hh.append(h)
fy, hh = np.array(fy), np.array(hh)
a, b = np.polyfit(fy, hh, 1)
for _ in range(5):
    r = np.abs(hh - (a * fy + b)); k = r < 2 * np.median(r) + 1e-6; a, b = np.polyfit(fy[k], hh[k], 1)
hor0, ch = -b / a, 1.75 / a
print(f"from the players: camera height {ch:.1f} m, horizon y={hor0:.0f}px")
cx, cy = W / 2, H / 2


def make_ground(fov, hor, k1=0.0):
    f = (W / 2) / math.tan(math.radians(fov / 2)); tilt = math.atan2(cy - hor, f)
    def ground(u, v):
        xn, yn = (u - cx) / f, (v - cy) / f
        sc = 1.0 / (1.0 + k1 * (xn * xn + yn * yn))                    # division model: undistort the point
        d = np.array([xn * sc, yn * sc, 1.0]); c, s = math.cos(tilt), math.sin(tilt)
        yw, zw = d[1] * c + d[2] * s, -d[1] * s + d[2] * c
        if yw <= 1e-6: return None
        t = ch / yw; return np.array([d[0] * t, zw * t])
    return ground


def ang(u, v): return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


from scipy.optimize import minimize
def cost(x):
    fov, hor, k1 = x
    if not (40 < fov < 150) or not (20 < hor < 400) or abs(k1) > 1.5: return 1e3
    g = make_ground(fov, hor, k1); P = [g(*p) for p in (FL, FR, RE, LL)]
    if any(p is None for p in P): return 1e3
    gFL, gFR, gRE, gLL = P
    df, dr, dl = gFR - gFL, gRE - gFR, gLL - gFL
    return ang(dr, dl) + abs(ang(df, dl) - 90) + abs(ang(df, dr) - 90)
best = None
rng = np.random.default_rng(0)
for _ in range(60):
    x0 = [rng.uniform(60, 130), rng.uniform(hor0 - 50, hor0 + 80), rng.uniform(-0.6, 0.6)]
    r = minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=600))
    if best is None or r.fun < best[0]: best = (r.fun, r.x)
e, (fov, hor, k1) = best
print(f"search done: lowest angle error {e:.2f} deg at FOV {fov:.0f}, horizon y={hor:.0f}px, lens distortion k1={k1:.2f}")
g = make_ground(fov, hor, k1); gFL, gFR, gRE, gLL = [g(*p) for p in (FL, FR, RE, LL)]
best = (e, fov, hor, gFL, gFR, gRE, gLL)
e, fov, hor, gFL, gFR, gRE, gLL = best
print(f"best fit: FOV {fov} deg, horizon y={hor}px  (angle errors total {e:.1f} deg)")
df, dr, dl = gFR - gFL, gRE - gFR, gLL - gFL
print(f"  far wall length {np.linalg.norm(df):.1f} m | angle between the two touchlines {ang(dr, dl):.1f} deg (0 = parallel) | far wall vs left line {ang(df, dl):.0f} deg, vs right line {ang(df, dr):.0f} deg (90 = square)")
u = df / np.linalg.norm(df)
width = abs(np.cross(u, gRE - gFL)) / 1 if False else np.linalg.norm(df)
print(f"  => pitch width about {width:.1f} m   (assumed ~27.5 m)")
# pitch frame: origin FL, x along the far wall toward FR, y toward the camera (perpendicular)
vy = np.array([-u[1], u[0]]); vy = vy if np.dot(vy, gLL - gFL) > 0 else -vy
g = make_ground(fov, hor, k1)
us, vs = np.meshgrid(np.linspace(150, 1880, 14), np.linspace(270, 1060, 12))
Pix = np.array([[x, y] for x, y in zip(us.ravel(), vs.ravel())], np.float32)
Gp = []
for x, y in Pix:
    q = g(x, y) - gFL; Gp.append([float(np.dot(q, u)), float(np.dot(q, vy))])
Gp = np.array(Gp, np.float32)
Hg, _ = cv2.findHomography(Gp, Pix, 0); np.save(D + "/H_pitch_to_img.npy", Hg)
back = cv2.perspectiveTransform(Gp.reshape(-1, 1, 2), Hg).reshape(-1, 2)
print(f"  homography saved (max refit error {np.abs(back - Pix).max():.2f} px). Pitch frame: x along the far wall (0..{width:.0f} m), y = metres toward the camera")
bd = {"far wall": [float(np.dot(g(*FL) - gFL, u)), float(np.dot(g(*FL) - gFL, vy)), float(np.dot(g(*FR) - gFL, u)), float(np.dot(g(*FR) - gFL, vy))]}
corners = {n: [float(np.dot(g(*p) - gFL, u)), float(np.dot(g(*p) - gFL, vy))] for n, p in (("FL", FL), ("FR", FR), ("RE", RE), ("LL", LL))}
json.dump(corners, open(D + "/boundary_m.json", "w")); print("boundary corners (m):", {k: [round(x, 1) for x in v] for k, v in corners.items()})
