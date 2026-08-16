#!/usr/bin/env python3
"""
Test Custom Video Runner
========================
Usage:
    .venv/bin/python scripts/test_my_video.py path/to/your_video.mp4

If no path is provided, it checks for any .mp4 file in `test_videos/`.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

# Default to SQLite for quick local test
if not os.getenv("DATABASE_URL") and not os.getenv("POSTGRES_SERVER"):
    os.environ["DATABASE_URL"] = "sqlite:////tmp/gaffer_test.db"

from app.db.database import engine, SessionLocal, Base
from app.db.models import Camera, Match, MatchStatus, Player, Detection, Event, PlayerStat
from app.tasks.pipeline import trigger_match_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("gaffer.test_video")


def main():
    # 1. Determine video file path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Check default test_videos directory
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_videos"))
        os.makedirs(default_dir, exist_ok=True)
        files = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith(".mp4")]
        if not files:
            print("=" * 70)
            print("❌ No video file specified!")
            print(f"👉 Please drop your test video in:\n   {default_dir}/test_match.mp4")
            print("   Or run:")
            print("   .venv/bin/python scripts/test_my_video.py /path/to/your_video.mp4")
            print("=" * 70)
            sys.exit(1)
        video_path = files[0]

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found.")
        sys.exit(1)

    print("=" * 70)
    print(f"⚽ Processing Video: {video_path}")
    print("=" * 70)

    # 2. Setup DB & Match session
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Get or create camera
    cam = db.query(Camera).first()
    if not cam:
        cam = Camera(
            serial_number="GAFFER-CAM-TEST",
            turf_name="Local Test Turf",
            api_key="gaffer_test_key_123",
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)

    match_title = os.path.splitext(os.path.basename(video_path))[0]
    match = Match(
        camera_id=cam.id,
        title=f"Test: {match_title}",
        video_s3_url=video_path,
        status=MatchStatus.PROCESSING,
    )
    db.add(match)
    db.commit()
    db.refresh(match)

    print(f"Created Match ID: {match.id}")
    print("Running YOLOv8 Detection + Centroid Tracking + Event Analysis...")

    # 3. Execute CV Pipeline
    result = trigger_match_pipeline(match_id=match.id, s3_url=video_path, run_async=False)

    # 4. Fetch & display results
    db.refresh(match)
    players = db.query(Player).filter(Player.match_id == match.id).all()
    detections = db.query(Detection).filter(Detection.match_id == match.id).all()
    events = db.query(Event).filter(Event.match_id == match.id).all()
    stats = db.query(PlayerStat).filter(PlayerStat.match_id == match.id).all()

    print("\n" + "=" * 70)
    print(f"📊 MATCH ANALYSIS COMPLETE: {match.title}")
    print(f"Duration: {match.duration_sec}s | Detections: {len(detections)} | Players Tracked: {len(players)}")
    print(f"Final Score: HOME {match.meta_data.get('score', {}).get('home', 0)} - {match.meta_data.get('score', {}).get('away', 0)} AWAY")
    print("=" * 70)

    print("\n🎯 DETECTED EVENTS (Goals, Passes, Possession Changes):")
    if events:
        for e in events:
            print(f"  • [{e.timestamp_ms / 1000.0:.1f}s] {e.event_type} (Team: {e.team}) - {json.dumps(e.data_json)}")
    else:
        print("  (No discrete goal/pass events detected in this clip window)")

    print("\n🏃 TOP PLAYERS BY DISTANCE & SPEED:")
    for s in sorted(stats, key=lambda x: x.distance_m, reverse=True)[:10]:
        p_name = s.player.player_name if s.player else "Player"
        p_team = s.player.team if s.player else "N/A"
        print(f"  • {p_name} ({p_team}): Dist={s.distance_m:.1f}m | Top Speed={s.top_speed_kmh:.1f}km/h | Passes={s.passes}")

    if stats and stats[0].heatmap_json:
        print("\n🗺️  PLAYER #1 HEATMAP (Pitch Density):")
        for row in stats[0].heatmap_json[:10]:
            line = "".join(["█" if v > 0.5 else "▒" if v > 0.2 else "·" for v in row[:30]])
            print(f"   |{line}|")

    print("\n" + "=" * 70)
    print(f"Results stored in SQLite: /tmp/gaffer_test.db")
    print("=" * 70)
    db.close()


if __name__ == "__main__":
    main()
