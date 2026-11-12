#!/usr/bin/env python3
"""
Label-free diagnostics for a fixed-camera clip: detect (3 native-resolution tiles
over the pitch band), track, split teams by shirt colour, follow the ball, count
passes. Uses NO dataset labels - everything below is computed from the video.

Usage: python clip_diagnostics.py <clip.mp4> <out_dir> [seconds=120] [stride=8] [band_y0=480]
Writes: frames.jsonl (per sampled frame), summary.json, check.webm (annotated).
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

VIDEO, OUT = sys.argv[1], Path(sys.argv[2])
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 120
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 8
BAND_Y0 = int(sys.argv[5]) if len(sys.argv) > 5 else 480
MODEL = r".local_data/models/football-player-detection.pt"
CONF, TILE = 0.25, 1280
BALL, GK, PLAYER, REF = 0, 1, 2, 3
POSSESS_RATIO = 1.2     # ball within 1.2 x player box height of their feet
MIN_HOLD = 2            # sampled frames
MAX_GAP_S = 3.0


def tiles_x(w):
    xs = list(range(0, max(1, w - TILE + 1), TILE))
    if xs[-1] != w - TILE:
        xs.append(w - TILE)
    return xs


def detect(model, frame):
    h, w = frame.shape[:2]
    y0 = min(BAND_Y0, max(0, h - TILE))
    outs = []
    for x0 in tiles_x(w):
        d = sv.Detections.from_ultralytics(
            model(frame[y0:y0 + TILE, x0:x0 + TILE], imgsz=TILE, verbose=False, conf=CONF)[0])
        d.xyxy = d.xyxy + np.array([x0, y0, x0, y0], dtype=np.float32)
        outs.append(d)
    d = sv.Detections.merge(outs)
    ball = d[d.class_id == BALL]
    rest = d[d.class_id != BALL].with_nms(threshold=0.5, class_agnostic=True)
    return rest, ball


def shirt_color(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    crop = frame[y1 + int(.2 * h):y1 + int(.55 * h), x1 + int((x2 - x1) * .2):x2 - int((x2 - x1) * .2)]
    if crop.size == 0:
        return None
    return crop.reshape(-1, 3).mean(0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(SECONDS * fps))
    tracker = sv.ByteTrack(frame_rate=max(1, int(fps / STRIDE)), track_activation_threshold=0.25,
                           lost_track_buffer=int(10 * fps / STRIDE), minimum_matching_threshold=0.8)
    rows, colors, t0 = [], {}, time.time()
    sampled = list(range(0, n_frames, STRIDE))
    for k, idx in enumerate(sampled):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        rest, ball = detect(model, frame)
        rest = tracker.update_with_detections(rest)
        tr = []
        for t, c, box, cf in zip(rest.tracker_id, rest.class_id, rest.xyxy, rest.confidence):
            col = shirt_color(frame, box) if c in (PLAYER, GK) else None
            if col is not None:
                colors.setdefault(int(t), []).append(col)
            tr.append([int(t), int(c), *[round(float(v), 1) for v in box], round(float(cf), 3)])
        bl = [[*[round(float(v), 1) for v in b], round(float(cf), 3)] for b, cf in zip(ball.xyxy, ball.confidence)]
        rows.append({"frame": idx, "tracks": tr, "ball": bl})
        if k % 10 == 0:
            el = time.time() - t0
            print(f"{k + 1}/{len(sampled)} frames, {el / (k + 1):.1f}s/frame, "
                  f"eta {el / (k + 1) * (len(sampled) - k - 1) / 60:.1f} min", flush=True)
    cap.release()

    # --- team split by shirt colour (per track, 2-means over track mean colours)
    ids = [i for i, v in colors.items() if len(v) >= 3]
    mean = np.array([np.mean(colors[i], 0) for i in ids], dtype=np.float32)
    team = {}
    if len(ids) >= 2:
        _, lab, _ = cv2.kmeans(mean, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                               5, cv2.KMEANS_PP_CENTERS)
        team = {i: int(l) for i, l in zip(ids, lab.ravel())}

    # --- ball: best-confidence candidate per frame
    ball_pts = {r["frame"]: max(r["ball"], key=lambda b: b[4]) for r in rows if r["ball"]}

    # --- possession (pixel space, calibration-free) and passes
    holder = []
    for r in rows:
        b = ball_pts.get(r["frame"])
        best = None
        if b:
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for t, c, x1, y1, x2, y2, _ in r["tracks"]:
                if c not in (PLAYER, GK):
                    continue
                fx, fy = (x1 + x2) / 2, y2
                d = np.hypot(bx - fx, by - fy) / max(y2 - y1, 1)
                if d < POSSESS_RATIO and (best is None or d < best[0]):
                    best = (d, t)
        holder.append(best[1] if best else None)
    runs, cur, start = [], None, 0
    for i, h in enumerate(holder + [None]):
        if h != cur:
            if cur is not None and i - start >= MIN_HOLD:
                runs.append((cur, rows[start]["frame"], rows[i - 1]["frame"]))
            cur, start = h, i
    events = []
    for a, b in zip(runs, runs[1:]):
        if a[0] == b[0] or (b[1] - a[2]) / fps > MAX_GAP_S:
            continue
        ta, tb = team.get(a[0]), team.get(b[0])
        kind = "pass" if ta is not None and ta == tb else "turnover" if ta is not None and tb is not None else "unknown"
        events.append({"frame": b[1], "t": round(b[1] / fps, 1), "from": a[0], "to": b[0], "kind": kind})

    # --- diagnostics
    life = {}
    for r in rows:
        for t, c, *_ in r["tracks"]:
            life[t] = life.get(t, 0) + 1
    pl = [sum(1 for t in r["tracks"] if t[1] in (PLAYER, GK)) for r in rows]
    summary = {
        "clip": VIDEO, "seconds": n_frames / fps, "stride": STRIDE, "sampled_frames": len(rows),
        "players_per_frame": {"mean": float(np.mean(pl)), "min": int(np.min(pl)), "max": int(np.max(pl))},
        "distinct_track_ids": len(life),
        "tracks_lasting_over_20s": sum(1 for v in life.values() if v * STRIDE / fps > 20),
        "tracks_under_2s": sum(1 for v in life.values() if v * STRIDE / fps < 2),
        "ball_detected_frames": len(ball_pts), "ball_detection_rate": len(ball_pts) / max(len(rows), 1),
        "frames_ball_near_a_player": sum(1 for h in holder if h is not None),
        "team_counts": {str(k): sum(1 for v in team.values() if v == k) for k in (0, 1)},
        "passes": sum(e["kind"] == "pass" for e in events),
        "turnovers": sum(e["kind"] == "turnover" for e in events),
        "unknown_team_events": sum(e["kind"] == "unknown" for e in events),
        "events": events,
        "seconds_per_frame": (time.time() - t0) / max(len(rows), 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(OUT / "frames.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (OUT / "teams.json").write_text(json.dumps({str(k): v for k, v in team.items()}))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    main()
