#!/usr/bin/env python3
"""Boxed video from identities.json (sequential frame reads).
python render_identities.py <video> <out_dir> <start_s> [width=1280] [legend_W=white shirts] [legend_B=black shirts]"""
import sys, os, json
import numpy as np
import cv2
video, out, st = sys.argv[1], sys.argv[2], float(sys.argv[3])
WOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
LW = sys.argv[5] if len(sys.argv) > 5 else "white shirts"; LB = sys.argv[6] if len(sys.argv) > 6 else "dark shirts"
meta = json.load(open(out + "/meta.json")); ids = json.load(open(out + "/identities.json"))
rows = sorted((json.loads(l)["frame"] for l in open(out + "/dets.jsonl")))
sfps = meta["fps"] / meta["stride"]; sc = WOUT / meta["width"]; H = int(meta["height"] * sc) // 2 * 2
cap = cv2.VideoCapture(video); cap.set(1, rows[0]); cur = rows[0]
wr = cv2.VideoWriter(out + "/identities_boxed.webm", cv2.VideoWriter_fourcc(*"VP80"), sfps, (WOUT, H))
col = {"W": (255, 255, 255), "B": (0, 140, 255), "?": (0, 255, 0)}
for i, fr in enumerate(rows):
    while cur < fr: cap.grab(); cur += 1
    ok, im = cap.read(); cur += 1
    if not ok: break
    im = cv2.resize(im, (WOUT, H), interpolation=cv2.INTER_AREA); vis = 0
    for k, I in ids.items():
        b = I["rows"].get(str(i))
        if b is None: continue
        vis += 1; x1, y1, x2, y2 = [int(v * sc) for v in b[:4]]; c = col.get(I["team"], (0, 255, 0))
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 1 if b[4] else 2); cv2.putText(im, I["label"], (x1, max(14, y1 - 4)), 0, 0.6, c, 2)
    cv2.rectangle(im, (0, 0), (760, 70), (0, 0, 0), -1)
    cv2.putText(im, f"t={st + i / sfps:5.1f}s   identified {vis}/{len(ids)}", (10, 28), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(im, f"{os.environ.get('LBL_W', 'W')} = {LW} (white boxes)    {os.environ.get('LBL_B', 'B')} = {LB} (orange boxes)", (10, 56), 0, 0.6, (200, 200, 200), 1)
    wr.write(im)
wr.release(); print("wrote", out + "/identities_boxed.webm")
