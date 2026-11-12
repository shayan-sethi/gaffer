#!/usr/bin/env python3
"""Camera from the players (height vs foot position) + projection of picked landmarks to ground metres.
python tolworth_calib.py <video> <out_dir> <start_s> <seconds> <n> [fov_deg=90]"""
import sys, os, json, math
import numpy as np
video, out, st, sec, n = sys.argv[1:6]
FOV = float(sys.argv[6]) if len(sys.argv) > 6 else 90.0
sys.argv = ["clip_ids.py", video, out, st, sec, n]
sys.path.insert(0, "scripts")
import clip_ids as C
meta = json.load(open(out + "/meta.json"))
rows = sorted((json.loads(l) for l in open(out + "/dets.jsonl")), key=lambda r: r["frame"]); sfps = meta["fps"] / meta["stride"]
W, H = meta["width"], meta["height"]
pf, _ = C.track(rows, np.ones((H, W), np.uint8), [], sfps)
ident, *_ = C.lock_identities(pf, sfps, colourer=C.TrackletColour())
fy, hh = [], []
for I in ident:
    for o in I["frames"].values():
        if not o.get("interp") and C.height(o) >= 40:
            fy.append(o["box"][3]); hh.append(C.height(o))
fy, hh = np.array(fy), np.array(hh)
a, b = np.polyfit(fy, hh, 1)
for _ in range(5):
    res = np.abs(hh - (a * fy + b)); keep = res < 2.0 * np.median(res) + 1e-6
    a, b = np.polyfit(fy[keep], hh[keep], 1)
horizon, cam_h = -b / a, 1.75 / a
f = (W / 2) / math.tan(math.radians(FOV / 2)); cx, cy = W / 2, H / 2
tilt = math.atan2(cy - horizon, f)
print(f"FOV {FOV:.0f}: horizon y={horizon:.0f}px, camera height {cam_h:.1f} m, tilt {math.degrees(tilt):.0f} deg, {keep.sum()} of {len(fy)} detections used")


def ground(u, v):
    d = np.array([(u - cx) / f, (v - cy) / f, 1.0]); c, s = math.cos(tilt), math.sin(tilt)
    y_w, z_w = d[1] * c + d[2] * s, -d[1] * s + d[2] * c
    t = cam_h / y_w
    return np.array([d[0] * t, z_w * t])


pts = {"far goal left post": (480, 290), "far goal right post": (645, 290), "far-left corner": (150, 257), "far-right corner": (857, 262), "near goal left post": (1775, 745)}
G = {k: ground(*v) for k, v in pts.items()}
for k, v in G.items(): print(f"  {k:22s} ground (across {v[0]:6.1f} m, away {v[1]:6.1f} m)")
d = lambda p, q: float(np.linalg.norm(G[p] - G[q]))
print(f"  far goal width            {d('far goal left post','far goal right post'):.1f} m   (a 5-a-side goal is usually 3.7-4.9 m)")
print(f"  far wall width (corner to corner) {d('far-left corner','far-right corner'):.1f} m   (assumed pitch width ~27.5 m)")
mid_far = (G["far goal left post"] + G["far goal right post"]) / 2
print(f"  far goal to the near goal's left post {float(np.linalg.norm(mid_far - G['near goal left post'])):.1f} m   (assumed pitch length ~36.5 m)")
