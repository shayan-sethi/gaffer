#!/usr/bin/env python3
"""Render a clip from identities.json (SAM 2 identities), ball_track.csv and pass_events.json: boxes coloured by team, labels, ball ring + trail, pass captions, running tally.
python render_sam2.py <out_dir> <n_rows> [outfile]"""
import sys, json, csv
from pathlib import Path
import numpy as np, cv2

D = Path(sys.argv[1]); N = int(sys.argv[2]); OUTF = Path(sys.argv[3]) if len(sys.argv) > 3 else D / "sam2_render.webm"
meta = json.load(open(D / "meta.json")); ids = json.load(open(D / "identities.json")); ev = json.load(open(D / "pass_events.json"))
Y0 = int(meta.get("band_y0", 0)); Y1 = int(meta.get("band_y1", meta["height"])); W = meta["width"]; S = 0.5; sfps = meta["fps"] / meta["stride"]
ball = []
for r in csv.DictReader(open(D / "ball_track.csv")):
    if r["x"] != "" and r["source"] in ("det", "interp"):
        ball.append((float(r["t"]), float(r["x"]), float(r["y"])))
bt = np.array([b[0] for b in ball]) if ball else np.zeros(0)
COL = {"W": (168, 95, 255), "B": (255, 168, 74), "R": (0, 120, 255), "Y": (60, 235, 205), "T": (255, 200, 80)}
PAL = [(255,120,120),(120,255,120),(120,120,255),(255,255,120),(255,120,255),(120,255,255),(0,160,255),(160,0,255),(255,160,0),(0,255,160),(200,200,200),(90,90,255),(60,200,60),(200,60,200)]
PC = {I["label"]: PAL[i % len(PAL)] for i, I in enumerate(ids.values())}
colof = lambda I: COL[I["team"]] if I["team"] in COL else PC[I["label"]]
cap = cv2.VideoCapture(meta["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, meta["start_frame"])
sys.path.insert(0, str(Path(__file__).resolve().parent)); import device
vw, OUTF = device.video_writer(OUTF, sfps, (int(W * S), int((Y1 - Y0) * S)))
trail = []
for r in range(N):
    ok, im = cap.read()
    if not ok:
        break
    for _ in range(meta["stride"] - 1):
        cap.grab()
    im = im[Y0:Y1].copy(); t = r / sfps
    for I in ids.values():
        b = I["rows"].get(str(r))
        if b:
            c = colof(I)
            cv2.rectangle(im, (int(b[0]), int(b[1] - Y0)), (int(b[2]), int(b[3] - Y0)), c, 2)
            cv2.putText(im, I["label"], (int(b[0]), int(b[1] - Y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    j = int(np.argmin(np.abs(bt - t))) if len(bt) else None
    if j is not None and abs(bt[j] - t) <= 0.1:
        trail.append((int(ball[j][1]), int(ball[j][2] - Y0)))
    else:
        trail.append(None)
    pts = [p for p in trail[-12:] if p]
    for a, b2 in zip(pts, pts[1:]):
        cv2.line(im, a, b2, (255, 255, 255), 1)
    if trail[-1]:
        cv2.circle(im, trail[-1], 12, (255, 255, 255), 2)
    shown = [e for e in ev if 0 <= t - e["t"] <= 2.0][-3:]
    for n, e in enumerate(shown):
        txt = f"{e['kind'].upper()}{'?' if e['confidence'] != 'seen' else ''}  {e['frm']} -> {e['to']}"
        cv2.putText(im, txt, (14, 60 + 34 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5); cv2.putText(im, txt, (14, 60 + 34 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    done = [e for e in ev if e["t"] <= t]
    tally = f"t={t:4.1f}s   passes so far: {sum(e['kind'] == 'pass' for e in done)}" + (f"   turnovers {sum(e['kind'] == 'turnover' for e in done)}" if any(I['team'] in COL for I in ids.values()) else "")
    cv2.putText(im, tally, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5); cv2.putText(im, tally, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    vw.write(cv2.resize(im, (int(W * S), int((Y1 - Y0) * S))))
vw.release(); print("wrote", OUTF)
