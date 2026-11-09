#!/usr/bin/env python3
"""Renders a check video: player boxes (id + voted team), tracked ball, current
possession, pass/turnover captions, frame + time. Usage: roboflow_verify_video.py <video> <pipeline_dir> <out.mp4>"""
import sys, os, json
import cv2
import numpy as np

VIDEO, D, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
cache = [json.loads(l) for l in open(os.path.join(D, "cache.jsonl"))]
clean = {r["frame"]: r for r in json.load(open(os.path.join(D, "tracks_clean.json")))}
ball = {o["frame"]: o for o in json.load(open(os.path.join(D, "ball_clean.json")))}
pj = json.load(open(os.path.join(D, "passes.json")))
runs, events = pj["possession_runs"], pj["events"]
COL = {0: (255, 255, 255), 1: (255, 160, 0)}  # BGR: team 0 white, team 1 blue-ish
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = 960, 540
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*("vp80" if OUT.endswith(".webm") else "mp4v")), fps, (W, H))
sx, sy = W / cap.get(cv2.CAP_PROP_FRAME_WIDTH), H / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
caption, caption_until = "", -1
for i in range(len(cache)):
    ok, img = cap.read()
    if not ok:
        break
    img = cv2.resize(img, (W, H))
    team = {t: c for t, c, _ in clean[i]["players"] + clean[i]["goalkeepers"]}
    holder = next((r["player"] for r in runs if r["start"] <= i <= r["end"]), None)
    for t, cls, x1, y1, x2, y2, _ in cache[i]["tracks"]:
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        if cls == 3:
            cv2.rectangle(img, p1, p2, (0, 215, 255), 1); continue
        c = COL.get(team.get(t), (150, 150, 150))
        cv2.rectangle(img, p1, p2, c, 3 if t == holder else 1)
        cv2.putText(img, f"#{t}", (p1[0], p1[1] - 4), 0, 0.5, c, 1, cv2.LINE_AA)
    b = ball.get(i)
    if b:
        cv2.circle(img, (int(b["px"][0] * sx), int(b["px"][1] * sy)), 14, (0, 0, 255) if b["interpolated"] else (0, 255, 0), 2)
    for e in events:
        if e["frame_release"] == i:
            caption = f"{e['kind'].upper()}: #{e['from']} -> #{e['to']} ({e['travel_m']} m)"
            caption_until = e["frame_receive"] + 40
    if i > caption_until:
        caption = ""
    cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(img, f"frame {i}  t={i/fps:.1f}s   ball: green=detected red=interpolated   thick box=has ball", (8, 23), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if caption:
        cv2.rectangle(img, (0, H - 44), (W, H), (0, 0, 0), -1)
        cv2.putText(img, caption, (12, H - 14), 0, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    out.write(img)
out.release()
