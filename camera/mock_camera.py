import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("camera.mock")


def record_mock_match(output_path: str, duration_sec: int = 5, fps: int = 30) -> str:
    """
    Simulates camera recording football turf footage to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Setup simulated players
    players = []
    for i in range(10):
        team = "HOME" if i < 5 else "AWAY"
        base_x = 300 + (i % 5) * 140 if team == "HOME" else 700 + (i % 5) * 110
        base_y = 150 + (i % 5) * 90
        color = (240, 50, 50) if team == "HOME" else (50, 50, 240)
        players.append({"base_x": base_x, "base_y": base_y, "color": color, "team": team})

    # Ball trajectory towards goal
    ball_start = (width // 2, height // 2)
    ball_end = (width - 40, height // 2)

    for f in range(total_frames):
        # Turf green background
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)

        # White pitch lines
        cv2.rectangle(frame, (60, 40), (width - 60, height - 40), (255, 255, 255), 3)
        cv2.line(frame, (width // 2, 40), (width // 2, height - 40), (255, 255, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 80, (255, 255, 255), 2)
        cv2.rectangle(frame, (60, height // 2 - 120), (160, height // 2 + 120), (255, 255, 255), 2)
        cv2.rectangle(frame, (width - 160, height // 2 - 120), (width - 60, height // 2 + 120), (255, 255, 255), 2)

        # Move players
        t = f / float(total_frames)
        for idx, p in enumerate(players):
            px = int(p["base_x"] + 35 * np.sin(2 * np.pi * t * 2 + idx))
            py = int(p["base_y"] + 25 * np.cos(2 * np.pi * t * 2 + idx))
            cv2.rectangle(frame, (px - 15, py - 40), (px + 15, py), p["color"], -1)
            cv2.circle(frame, (px, py - 48), 10, (220, 220, 220), -1)

        # Move ball
        bx = int(ball_start[0] + (ball_end[0] - ball_start[0]) * t)
        by = int(ball_start[1] + (ball_end[1] - ball_start[1]) * t + 35 * np.sin(np.pi * t))
        cv2.circle(frame, (bx, by), 8, (255, 255, 255), -1)
        cv2.circle(frame, (bx, by), 8, (0, 0, 0), 1)

        # Overlay timecode
        cv2.putText(
            frame,
            f"GAFFER REC: {f/fps:.1f}s / {duration_sec}s",
            (80, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        out.write(frame)

    out.release()
    logger.info(f"Recorded mock match video: {output_path} ({total_frames} frames)")
    return output_path
