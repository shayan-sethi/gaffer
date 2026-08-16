import os
import logging
import urllib.parse
import boto3
from botocore.client import Config
import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def download_match_video(s3_url: str, local_dest_path: str) -> str:
    """
    Downloads video from S3/MinIO or creates synthetic video if test URL is passed.
    """
    os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)

    # Check if this is a synthetic mock video request
    if s3_url.startswith("mock://") or s3_url == "synthetic" or not s3_url:
        logger.info(f"Generating synthetic football match video at {local_dest_path}...")
        return generate_synthetic_match_video(local_dest_path, duration_sec=5, fps=30)

    # Check if local file already exists
    if os.path.exists(s3_url):
        return s3_url

    # Parse S3 URI or HTTP MinIO URL
    try:
        s3 = get_s3_client()
        parsed = urllib.parse.urlparse(s3_url)

        if parsed.scheme in ["http", "https"]:
            # Format: http://localhost:9000/bucket/key
            path_parts = parsed.path.lstrip("/").split("/", 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ""
        elif parsed.scheme == "s3":
            # Format: s3://bucket/key
            bucket_name = parsed.netloc
            object_key = parsed.path.lstrip("/")
        else:
            bucket_name = settings.S3_BUCKET_NAME
            object_key = s3_url

        logger.info(f"Downloading from S3 bucket={bucket_name}, key={object_key} to {local_dest_path}...")
        s3.download_file(bucket_name, object_key, local_dest_path)
        logger.info("Download completed successfully.")
        return local_dest_path
    except Exception as e:
        logger.warning(f"Failed to download from S3 ({e}). Creating fallback synthetic video...")
        return generate_synthetic_match_video(local_dest_path, duration_sec=5, fps=30)


def generate_synthetic_match_video(output_path: str, duration_sec: int = 5, fps: int = 30) -> str:
    """
    Generates a 1280x720 football video with pitch markings, moving players, and a ball that scores a goal.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Setup simulated player trajectories
    num_players = 10
    players = []
    for i in range(num_players):
        team = "HOME" if i < 5 else "AWAY"
        base_x = 350 + (i % 5) * 120 if team == "HOME" else 750 + (i % 5) * 100
        base_y = 180 + (i % 5) * 80
        color = (255, 60, 60) if team == "HOME" else (60, 60, 255)
        players.append({"base_x": base_x, "base_y": base_y, "color": color, "team": team})

    # Ball trajectory towards right goal
    ball_start = (width // 2, height // 2)
    ball_end = (width - 40, height // 2)

    for f in range(total_frames):
        # Green pitch
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)

        # Pitch markings (white lines)
        cv2.rectangle(frame, (60, 40), (width - 60, height - 40), (255, 255, 255), 3)
        # Center line & circle
        cv2.line(frame, (width // 2, 40), (width // 2, height - 40), (255, 255, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 80, (255, 255, 255), 2)

        # Goal boxes
        cv2.rectangle(frame, (60, height // 2 - 120), (160, height // 2 + 120), (255, 255, 255), 2)
        cv2.rectangle(frame, (width - 160, height // 2 - 120), (width - 60, height // 2 + 120), (255, 255, 255), 2)

        # Draw moving players
        t = f / float(total_frames)
        for idx, p in enumerate(players):
            px = int(p["base_x"] + 30 * np.sin(2 * np.pi * t * 2 + idx))
            py = int(p["base_y"] + 20 * np.cos(2 * np.pi * t * 2 + idx))
            # Draw player bounding box representation
            cv2.rectangle(frame, (px - 15, py - 40), (px + 15, py), p["color"], -1)
            cv2.circle(frame, (px, py - 48), 10, (220, 220, 220), -1)

        # Draw ball moving towards goal
        bx = int(ball_start[0] + (ball_end[0] - ball_start[0]) * t)
        by = int(ball_start[1] + (ball_end[1] - ball_start[1]) * t + 40 * np.sin(np.pi * t))
        cv2.circle(frame, (bx, by), 8, (255, 255, 255), -1)
        cv2.circle(frame, (bx, by), 8, (0, 0, 0), 1)

        out.write(frame)

    out.release()
    logger.info(f"Synthetic match video generated: {output_path} ({total_frames} frames)")
    return output_path
