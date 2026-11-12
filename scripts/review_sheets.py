#!/usr/bin/env python3
"""Review sheets for pass/turnover events: for each event three crops (0.5 s before the passer's last touch, the handover, 0.5 s after) around the ball,
boxes of the two players drawn. The user marks each event right / wrong / missing-nearby; those marks are the ground truth for scoring any pass model.
python review_sheets.py <video> <out_dir> <start_s> [per_sheet=6]   -> <out_dir>/review/sheet_N.jpg and events_list.txt"""
import sys, json, csv
from pathlib import Path
import numpy as np, cv2

video, D, start = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3]); PER = int(sys.argv[4]) if len(sys.argv) > 4 else 6
meta = json.load(open(D / "meta.json")); ev = json.load(open(D / "pass_events.json")); ids = json.load(open(D / "identities.json"))
fps = meta["fps"]; sfps = fps / meta["stride"]; first = meta["start_frame"]
byl = {I["label"]: {int(k): v for k, v in I["rows"].items()} for I in ids.values()}
ball = {}
for r in csv.DictReader(open(D / "ball_track.csv")):
    if r["x"] != "" and r["source"] in ("det", "interp"):
        ball[round(float(r["t"]), 2)] = (float(r["x"]), float(r["y"]))
bt = np.array(sorted(ball))
cap = cv2.VideoCapture(video)
out = D / "review"; out.mkdir(exist_ok=True)
CW, CH = 560, 330


def ball_at(t):
    if not len(bt):
        return None
    j = int(np.argmin(np.abs(bt - t)))
    return ball[float(bt[j])] if abs(bt[j] - t) <= 0.5 else None


def crop(t, e, cx, cy):
    cap.set(cv2.CAP_PROP_POS_FRAMES, first + int(round(t * fps))); ok, im = cap.read()
    if not ok:
        return np.zeros((CH, CW, 3), np.uint8)
    x0 = int(min(max(0, cx - CW / 2), im.shape[1] - CW)); y0 = int(min(max(0, cy - CH / 2), im.shape[0] - CH)); c = im[y0:y0 + CH, x0:x0 + CW].copy()
    row = int(round(t * sfps))
    for lbl, col in ((e["frm"], (0, 200, 255)), (e["to"], (60, 255, 60))):
        b = byl[lbl].get(row) or byl[lbl].get(row - 1) or byl[lbl].get(row + 1)
        if b:
            cv2.rectangle(c, (int(b[0] - x0), int(b[1] - y0)), (int(b[2] - x0), int(b[3] - y0)), col, 2)
            cv2.putText(c, lbl, (int(b[0] - x0), int(b[1] - y0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    bp = ball_at(t)
    if bp:
        cv2.circle(c, (int(bp[0] - x0), int(bp[1] - y0)), 14, (255, 255, 255), 2)
    cv2.putText(c, f"{t + start:5.2f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return c


lines = []
for s in range(0, len(ev), PER):
    rows = []
    for n, e in enumerate(ev[s:s + PER], start=s + 1):
        t0 = e["t"]
        a, b = ball_at(max(0, t0 - 0.5)), ball_at(t0)
        cx, cy = ((a or b or (960, 600))[0] + (b or a or (960, 600))[0]) / 2, ((a or b or (960, 600))[1] + (b or a or (960, 600))[1]) / 2
        tiles = [crop(max(0, t0 - 0.6), e, cx, cy), crop(t0, e, cx, cy), crop(t0 + 0.6, e, cx, cy)]
        strip = np.hstack(tiles)
        cv2.rectangle(strip, (0, CH - 30), (strip.shape[1], CH), (0, 0, 0), -1)
        cv2.putText(strip, f"#{n}  {e['kind'].upper()} {e['frm']} -> {e['to']}  ({e['confidence']}, ball hidden {e['gap_s']} s)  [before | handover | after]", (6, CH - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(strip)
        lines.append(f"#{n}  {e['t']:.1f}s  {e['kind']}  {e['frm']} -> {e['to']}  {e['confidence']}")
    cv2.imwrite(str(out / f"sheet_{s // PER + 1}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
open(out / "events_list.txt", "w").write("\n".join(lines))
print(len(ev), "events ->", len(range(0, len(ev), PER)), "sheets in", out)
