#!/usr/bin/env python3
"""
One sequential pass over the video: add a grass-free shirt colour (Lab: L, a*1.5, b*1.5) to every detection in dets.jsonl
(3 extra numbers per detection; -999 = not enough non-grass pixels). After this, identity locking and team checks never
need to touch the video again. Sequential reading is MUCH faster than seeking on a large video.
  python clip_colours.py <out_dir>
"""
import sys, json, shutil
from pathlib import Path
import numpy as np
import cv2
import os
BIB_S = int(os.environ.get("BIB_S", 110)); BIB_V = int(os.environ.get("BIB_V", 110))      # a bib needs at least this saturation / brightness (fog washes out far bibs: use ~70/100)
OUT = Path(sys.argv[1])
meta = json.load(open(OUT / "meta.json"))
rows = [json.loads(l) for l in open(OUT / "dets.jsonl")]
if not (OUT / "dets_nolab.jsonl").exists():
    shutil.copy(OUT / "dets.jsonl", OUT / "dets_nolab.jsonl")
    rows = [json.loads(l) for l in open(OUT / "dets_nolab.jsonl")]
else:
    rows = [json.loads(l) for l in open(OUT / "dets_nolab.jsonl")]       # always start from the original
by = {r["frame"]: r for r in rows}
last = max(by)
cap = cv2.VideoCapture(meta["video"])
cap.set(cv2.CAP_PROP_POS_FRAMES, min(by))
fi = min(by)


def lab_feature(rgb):
    px = np.uint8([[[min(255, max(0, round(float(v)))) for v in rgb[::-1]]]])
    L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return [round(L * 100 / 255, 2), round(1.5 * (a - 128), 2), round(1.5 * (b - 128), 2)]


n = 0
while fi <= last:
    if fi in by:
        ok, im = cap.read()
        if not ok:
            break
        for d in by[fi]["dets"]:
            x1, y1, x2, y2 = [int(v) for v in d[:4]]
            h, w = y2 - y1, x2 - x1
            lab = [-999, -999, -999]; t = np.zeros(0); hsv = None
            if h >= 22:
                t = im[max(0, y1 + int(.18 * h)):y1 + int(.55 * h), max(0, x1 + int(.2 * w)):x2 - int(.2 * w)]
                if t.size:
                    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                    px = t.reshape(-1, 3)[~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 95) & (hsv[:, 1] > 35))]
                    if len(px) >= 6:
                        lab = lab_feature(np.median(px, axis=0)[::-1])
            sat = [-1.0] * 6                                # share of the torso that is a strong colour, split into 6 hue bins of 30 degrees (OpenCV hue 0..180)
            if h >= 22 and t.size:
                sm = (hsv[:, 1] >= BIB_S) & (hsv[:, 2] >= BIB_V)
                sat = [round(float((sm & (hsv[:, 0] >= 30 * k) & (hsv[:, 0] < 30 * k + 30)).mean()), 3) for k in range(6)]
            d[:] = d[:8] + lab + sat
        n += 1
        if n % 50 == 0:
            print(n, "frames", flush=True)
    else:
        if not cap.grab():
            break
    fi += 1
with open(OUT / "dets.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("done:", n, "frames with colours")
