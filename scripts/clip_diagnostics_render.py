#!/usr/bin/env python3
"""
Draws the results of clip_diagnostics.py on the clip: boxes round every tracked
player (colour = team split, label = track id), a ring on the ball candidate, and
captions for pass / turnover events. Crops to the pitch band and downsizes.

Usage: python clip_diagnostics_render.py <clip.mp4> <diag_dir> <out.webm> [band_y0=480] [out_fps=3]
"""
import sys, json
from pathlib import Path
import cv2

VIDEO, DIAG, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
BAND_Y0 = int(sys.argv[4]) if len(sys.argv) > 4 else 480
OUT_FPS = float(sys.argv[5]) if len(sys.argv) > 5 else 3
BAND_H, OUT_W = 1280, 1920
TEAM_COLOR = {0: (255, 140, 0), 1: (0, 200, 255), None: (200, 200, 200)}   # BGR
PLAYER, GK, REF = 2, 1, 3


def main():
    rows = [json.loads(l) for l in open(DIAG / "frames.jsonl")]
    teams = {int(k): v for k, v in json.load(open(DIAG / "teams.json")).items()}
    summary = json.load(open(DIAG / "summary.json"))
    events = summary["events"]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scale = OUT_W / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    out_h = int(BAND_H * scale)
    w = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"VP80"), OUT_FPS, (OUT_W, out_h))
    for r in rows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[BAND_Y0:BAND_Y0 + BAND_H].copy()
        for t, c, x1, y1, x2, y2, cf in r["tracks"]:
            col = (60, 60, 255) if c == REF else (0, 255, 0) if c == GK else TEAM_COLOR[teams.get(t)]
            p1, p2 = (int(x1), int(y1 - BAND_Y0)), (int(x2), int(y2 - BAND_Y0))
            cv2.rectangle(frame, p1, p2, col, 3)
            cv2.putText(frame, str(t), (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        for x1, y1, x2, y2, cf in r["ball"]:
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2 - BAND_Y0)), 28, (0, 0, 255), 4)
        recent = [e for e in events if 0 <= (r["frame"] - e["frame"]) / fps <= 2.0]
        for i, e in enumerate(recent):
            cv2.putText(frame, f"{e['kind'].upper()} {e['from']}->{e['to']}", (30, 70 + 70 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 5)
        cv2.putText(frame, f"t={r['frame'] / fps:5.1f}s  players tracked: "
                    f"{sum(1 for t in r['tracks'] if t[1] in (PLAYER, GK))}", (30, BAND_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        w.write(cv2.resize(frame, (OUT_W, out_h), interpolation=cv2.INTER_AREA))
    w.release()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
