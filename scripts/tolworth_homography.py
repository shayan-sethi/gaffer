#!/usr/bin/env python3
"""Ground-plane homography from the people-based camera model with a chosen field of view (no lens model), plus the user's boundary
lines projected through it.   python tolworth_homography.py [fov_deg=110]   -> H_pitch_to_img.npy, boundary_m.json"""
import sys, json, math
import numpy as np, cv2
FOV = float(sys.argv[1]) if len(sys.argv) > 1 else 110.0
D = ".local_data/tolworth_1080d"
meta = json.load(open(D + "/meta.json")); W, H = meta["width"], meta["height"]
FL, FR, RE, LL = np.load(".local_data/tolworth_user_lines.npy")
ids = json.load(open(D + "/identities.json"))
fy, hh = [], []
for I in ids.values():
    for b in I["rows"].values():
        h = b[3] - b[1]
        if not b[4] and h >= 40: fy.append(b[3]); hh.append(h)
fy, hh = np.array(fy), np.array(hh); a, b = np.polyfit(fy, hh, 1)
for _ in range(5):
    r = np.abs(hh - (a * fy + b)); k = r < 2 * np.median(r) + 1e-6; a, b = np.polyfit(fy[k], hh[k], 1)
hor, ch = -b / a, 1.75 / a; f = (W / 2) / math.tan(math.radians(FOV / 2)); cx, cy = W / 2, H / 2; tilt = math.atan2(cy - hor, f)


def ground(u, v):
    d = np.array([(u - cx) / f, (v - cy) / f, 1.0]); c, s = math.cos(tilt), math.sin(tilt)
    yw, zw = d[1] * c + d[2] * s, -d[1] * s + d[2] * c; t = ch / yw
    return np.array([d[0] * t, zw * t])


gFL = ground(*FL)
def ang(u, v): return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
P = {n: ground(*p) for n, p in (("FL", FL), ("FR", FR), ("RE", RE), ("LL", LL))}
df, dr, dl = P["FR"] - P["FL"], P["RE"] - P["FR"], P["LL"] - P["FL"]
print(f"FOV {FOV:.0f}, camera height {ch:.1f} m: far wall {np.linalg.norm(df):.1f} m | touchlines off parallel by {ang(dr, dl):.0f} deg | far wall vs left {ang(df, dl):.0f} deg, vs right {ang(df, dr):.0f} deg")
us, vs = np.meshgrid(np.linspace(150, 1880, 14), np.linspace(270, 1060, 12)); Pix = np.array([[x, y] for x, y in zip(us.ravel(), vs.ravel())], np.float32)
G = np.array([ground(x, y) - gFL for x, y in Pix], np.float32)               # pitch frame: x = across, y = away from the camera, origin at the far-left corner
Hg, _ = cv2.findHomography(G, Pix, 0); np.save(D + "/H_pitch_to_img.npy", Hg)
json.dump({n: [float(v[0] - gFL[0]), float(v[1] - gFL[1])] for n, v in P.items()}, open(D + "/boundary_m.json", "w"))
print("saved H_pitch_to_img.npy and boundary_m.json")
