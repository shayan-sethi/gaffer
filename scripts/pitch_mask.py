#!/usr/bin/env python3
"""
Automatic pitch mask for a fixed camera: finds the green turf and keeps only detections
whose feet stand on it. No hand-drawn polygon. Removes people printed on advertising
boards, spectators behind fences, people beside the pitch.

  mask = build_mask(frames)            # list of BGR frames (players move, turf doesn't)
  keep = feet_on_pitch(mask, box)      # box = x1,y1,x2,y2

Demo: python pitch_mask.py <video> <out_prefix> <t1,t2,...>
"""
import sys
import numpy as np
import cv2


def green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # turf: green hue, reasonably saturated; shadows are darker but still green
    return ((h >= 35) & (h <= 90) & (s >= 60) & (v >= 30)).astype(np.uint8)


def build_mask(frames):
    """Per-pixel majority over several frames removes players; keep the biggest green blob."""
    votes = sum(green_mask(f) for f in frames)
    m = (votes * 2 > len(frames)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    # fill holes (players standing on the turf punch holes in the mask)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled


def feet_on_pitch(mask, box, margin=8):
    """True if the turf lies just below the box's feet (probes a few px under the bottom edge,
    left / centre / right). Players on the turf always have turf under their boots; people
    beyond the touchline or fence do not."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    H, W = mask.shape
    y = min(H - 1, int(y2) + margin)
    xs = [min(W - 1, max(0, cx + d)) for d in (-int((x2 - x1) / 4), 0, int((x2 - x1) / 4))]
    return sum(int(mask[y, x]) for x in xs) >= 2


def static_spots(boxes_per_frame, frac=0.4, tol=10):
    """Fixed camera: a 'person' that sits on the same spot (+-tol px) in >= frac of sampled
    frames spread over minutes is a printed figure (advert board, poster, statue), not a player.
    Returns [(cx, foot_y)] to exclude. Sample frames at least ~10 s apart so a standing
    keeper doesn't qualify."""
    pts = [(i, (b[0] + b[2]) / 2, b[3]) for i, bs in enumerate(boxes_per_frame) for b in bs]
    used, out = set(), []
    for k, (i, x, y) in enumerate(pts):
        if k in used:
            continue
        mem = [j for j, (_, x2, y2) in enumerate(pts) if abs(x2 - x) < tol and abs(y2 - y) < tol and j not in used]
        used.update(mem)
        if len({pts[j][0] for j in mem}) >= frac * len(boxes_per_frame):
            out.append((x, y))
    return out


def is_static(box, spots, tol=14):
    cx, fy = (box[0] + box[2]) / 2, box[3]
    return any(abs(cx - x) < tol and abs(fy - y) < tol for x, y in spots)


if __name__ == "__main__":
    from ultralytics import YOLO
    video, prefix = sys.argv[1], sys.argv[2]
    times = [float(t) for t in sys.argv[3].split(",")]
    cap = cv2.VideoCapture(video)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        frames.append(f)
    mask = build_mask(frames)
    model = YOLO("yolov8n.pt")
    for t, f in zip(times, frames):
        r = model(f, imgsz=1280, conf=0.15, classes=[0], verbose=False)[0]
        out = f.copy()
        ov = out.copy()
        ov[mask > 0] = (0, 200, 0)
        out = cv2.addWeighted(ov, 0.25, out, 0.75, 0)
        kept = rej = 0
        for b in r.boxes.xyxy.cpu().numpy():
            ok = feet_on_pitch(mask, b)
            kept += ok
            rej += not ok
            col = (0, 255, 0) if ok else (0, 0, 255)
            cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3)
        cv2.putText(out, f"t={t:.0f}s kept {kept} (green)  rejected {rej} (red)", (30, 60), 0, 1.6, (255, 255, 255), 4)
        cv2.imwrite(f"{prefix}_{int(t)}.jpg", out)
        print(t, "kept", kept, "rejected", rej)
    cv2.imwrite(f"{prefix}_mask.jpg", mask * 255)
