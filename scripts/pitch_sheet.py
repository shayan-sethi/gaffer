#!/usr/bin/env python3
"""Contact sheet of full-pitch (crowd cropped out) frames at a fixed time
interval, sized for legible manual review - no automated ball-position
assist (that proved unreliable), just plain sequential frames.
Usage: python pitch_sheet.py <video_path> <start_sec> <duration_sec> <interval_sec> <out_path> <cols>
"""
import sys
import cv2
import numpy as np


def main():
    video_path, start_sec, duration_sec, interval_sec, out_path, cols = sys.argv[1:7]
    start_sec, duration_sec, interval_sec, cols = float(start_sec), float(duration_sec), float(interval_sec), int(cols)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    times = np.arange(start_sec, start_sec + duration_sec, interval_sec)
    thumb_w, thumb_h = 640, 400  # crowd-cropped crop resized to this
    rows = int(np.ceil(len(times) / cols))
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, t in enumerate(times):
        frame_no = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue
        h = frame.shape[0]
        pitch_only = frame[int(h * 0.14):, :]  # drop crowd/stands band
        thumb = cv2.resize(pitch_only, (thumb_w, thumb_h))
        cv2.putText(thumb, f"{t:.1f}s", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        r, c = divmod(i, cols)
        sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {out_path} ({cols}x{rows} grid, {len(times)} frames, {start_sec:.1f}-{start_sec+duration_sec:.1f}s)")


if __name__ == "__main__":
    main()
