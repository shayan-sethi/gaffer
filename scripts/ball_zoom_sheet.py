#!/usr/bin/env python3
"""Build a contact sheet of zoomed-in, ball-centered crops for manual pass
counting - uses the (comparatively reliable) tracked ball position to know
where to zoom, rather than trusting per-player identity/team labels.
Usage: python ball_zoom_sheet.py <pickle_path> <video_path> <start_sec> <end_sec> <interval_sec> <out_path> <fps>
"""
import sys
import pickle
import cv2
import numpy as np


def main():
    pickle_path, video_path, start_sec, end_sec, interval_sec, out_path, fps = sys.argv[1:8]
    start_sec, end_sec, interval_sec, fps = float(start_sec), float(end_sec), float(interval_sec), float(fps)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    ball = next(t for t in data["tracks"].values() if t["class_name"] == "ball")
    ball_hist = ball["history"]  # [(frame_no, [x, y]), ...]
    ball_frames = np.array([h[0] for h in ball_hist])
    ball_pos = np.array([h[1] for h in ball_hist])

    cap = cv2.VideoCapture(video_path)

    times = np.arange(start_sec, end_sec, interval_sec)
    crop_w, crop_h = 560, 340
    cols = 4
    rows = int(np.ceil(len(times) / cols))
    sheet = np.zeros((rows * crop_h, cols * crop_w, 3), dtype=np.uint8)

    for i, t in enumerate(times):
        target_frame = t * fps
        idx = int(np.argmin(np.abs(ball_frames - target_frame)))
        frame_no = int(ball_frames[idx])
        bx, by = ball_pos[idx]
        gap = abs(ball_frames[idx] - target_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        ret, frame = cap.read()
        r, c = divmod(i, cols)
        if not ret:
            continue
        x1, y1 = int(bx - crop_w / 2), int(by - crop_h / 2)
        x1 = max(0, min(x1, frame.shape[1] - crop_w))
        y1 = max(0, min(y1, frame.shape[0] - crop_h))
        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        if crop.shape[:2] != (crop_h, crop_w):
            padded = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        label = f"{t:.1f}s"
        if gap > fps * 0.75:  # ball position more than ~0.75s stale
            label += " (stale)"
        cv2.putText(crop, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # crosshair on the tracked ball position
        cx, cy = crop_w // 2, crop_h // 2
        cv2.drawMarker(crop, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        sheet[r*crop_h:(r+1)*crop_h, c*crop_w:(c+1)*crop_w] = crop

    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Saved {out_path} ({cols}x{rows} grid, {len(times)} samples, {start_sec:.1f}-{end_sec:.1f}s)")


if __name__ == "__main__":
    main()
