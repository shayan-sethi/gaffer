#!/usr/bin/env python3
"""
Extract Ball-Labeling Candidate Frames
=======================================
Finds frames from a processed match video where the ball detector missed or
only weakly detected the ball, and saves them as JPGs for manual annotation
(e.g. upload to Roboflow) to improve ball recall via fine-tuning.

Usage:
    .venv/bin/python scripts/extract_labeling_candidates.py <video_path> [match_id] [--max N]

If match_id is omitted, uses the most recently created match in the local DB.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".local_data"))
DB_PATH = os.path.join(LOCAL_DATA_DIR, "gaffer_test.db")
if not os.getenv("DATABASE_URL") and not os.getenv("POSTGRES_SERVER"):
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"

import cv2
from app.db.database import SessionLocal
from app.db.models import Match, Detection

OUTPUT_DIR = os.path.join(LOCAL_DATA_DIR, "labeling_candidates")

# A "gap" is a run of consecutive frames with zero ball detections. Short gaps
# (a couple frames of motion blur/occlusion) aren't worth labeling - they're
# already bridged fine by interpolation. Long gaps are where the detector is
# actually failing and where new labeled examples help most.
MIN_GAP_FRAMES = 10
# How many frames to sample out of each qualifying gap (spread across it,
# not just the edges) - the ball is usually somewhere mid-flight/occluded
# for most of a gap, which is exactly the hard case we want examples of.
SAMPLES_PER_GAP = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("match_id", nargs="?", default=None)
    parser.add_argument("--max", type=int, default=250, help="Max total frames to extract")
    parser.add_argument("--min-gap", type=int, default=MIN_GAP_FRAMES)
    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    db = SessionLocal()
    if args.match_id:
        match = db.query(Match).filter(Match.id == args.match_id).first()
    else:
        match = db.query(Match).order_by(Match.created_at.desc()).first()
    if not match:
        print("No match found in local DB. Run scripts/test_my_video.py first.")
        sys.exit(1)
    print(f"Using match {match.id} ({match.title})")

    ball_frames = sorted(set(
        row[0] for row in db.query(Detection.frame_no)
        .filter(Detection.match_id == match.id, Detection.class_name == "ball")
        .all()
    ))
    if not ball_frames:
        print("No ball detections recorded for this match.")
        sys.exit(1)

    max_frame = db.query(Detection.frame_no).filter(Detection.match_id == match.id).order_by(Detection.frame_no.desc()).first()[0]
    print(f"Ball detected in {len(ball_frames)} / {max_frame + 1} frames ({100 * len(ball_frames) / (max_frame + 1):.1f}% recall)")

    # Find gaps: stretches of frame numbers with no ball detection.
    gaps = []
    prev = -1
    for f in ball_frames + [max_frame + 1]:
        gap_len = f - prev - 1
        if gap_len >= args.min_gap:
            gaps.append((prev + 1, f - 1))  # inclusive range with no ball detection
        prev = f

    total_gap_frames = sum(b - a + 1 for a, b in gaps)
    print(f"Found {len(gaps)} gaps >= {args.min_gap} frames (covering {total_gap_frames} frames total)")

    # Sort gaps longest-first so the most valuable (hardest) cases get priority
    # if --max forces us to cut the list short.
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)

    candidate_frames = []
    for start, end in gaps:
        length = end - start + 1
        n = min(SAMPLES_PER_GAP, length)
        for i in range(n):
            frame_no = start + round((i + 1) * length / (n + 1))
            candidate_frames.append(frame_no)
        if len(candidate_frames) >= args.max:
            break

    candidate_frames = sorted(set(candidate_frames))[:args.max]
    print(f"Selected {len(candidate_frames)} candidate frames to extract")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    saved = 0
    for frame_no in candidate_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            print(f"  ! Could not read frame {frame_no}, skipping")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_no:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()
    print(f"\nSaved {saved} frames to {OUTPUT_DIR}")
    print("Next: label the ball's bounding box in each of these (e.g. upload to Roboflow),")
    print("then merge the new labels into the training set and re-run fine-tuning.")


if __name__ == "__main__":
    main()
