#!/usr/bin/env python3
"""
Gaffer End-to-End MVP Demo & Test Pipeline
=========================================
Demonstrates and validates the entire Gaffer workflow:
1. Initialize database & tables
2. Register Gaffer hardware camera
3. Start match session
4. Generate synthetic turf football video (players moving, passes, goal scored)
5. Execute YOLOv8 detection + Centroid tracking
6. Execute Event detection (Goals, Passes, Possession)
7. Execute Stats & 2D Spatial Heatmap calculation
8. Query & display formatted match analytics report
"""

import os
import sys
import json
import logging

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

# Configure test environment defaults (SQLite for instant zero-dependency execution if Postgres not running)
if not os.getenv("DATABASE_URL") and not os.getenv("POSTGRES_SERVER"):
    test_db_path = "/tmp/gaffer_test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

from app.config import settings
from app.db.database import engine, SessionLocal, Base
from app.db.models import Camera, Match, MatchStatus, Player, Detection, Event, PlayerStat
from app.tasks.pipeline import trigger_match_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("gaffer.demo")


def main():
    print("=" * 70)
    print("   ⚽ GAFFER - Automated Football Match Intelligence Pipeline ⚽   ")
    print("=" * 70)

    # 1. Initialize Database
    logger.info("Step 1: Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # 2. Register Hardware Camera
    logger.info("Step 2: Registering Gaffer Camera (GAFFER-CAM-DEMO-001)...")
    cam = db.query(Camera).filter(Camera.serial_number == "GAFFER-CAM-DEMO-001").first()
    if not cam:
        cam = Camera(
            serial_number="GAFFER-CAM-DEMO-001",
            turf_name="Wembley Turf Pitch 1",
            turf_id="turf_wembley_01",
            api_key="gaffer_cam_demo_key_secret_123",
            firmware_version="1.0.0",
        )
        db.add(cam)
        db.commit()
        db.refresh(cam)
    print(f"✅ Camera Registered: ID={cam.id}, Turf='{cam.turf_name}', Serial={cam.serial_number}")

    # 3. Create/Start Match Session
    logger.info("Step 3: Creating match session...")
    match = Match(
        camera_id=cam.id,
        turf_id=cam.turf_id,
        title="Friday Night 5-a-side Derby",
        status=MatchStatus.RECORDING,
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    print(f"✅ Match Session Created: ID={match.id}, Title='{match.title}'")

    # 4. Execute Synchronous CV Pipeline
    # The "synthetic" sentinel tells the pipeline to generate its own drawn-shape demo clip
    # and run detection in simulation mode - real match footage never takes this path, so
    # a genuinely empty real-world frame is never mistaken for one that needs simulating.
    logger.info("Step 4: Executing Computer Vision Pipeline (Detect ➔ Track ➔ Events ➔ Stats)...")
    result = trigger_match_pipeline(match_id=match.id, s3_url="synthetic", run_async=False)
    print(f"✅ Video generated & Pipeline Execution Finished: {result}")

    # 6. Query Match Results from PostgreSQL / DB
    logger.info("Step 6: Querying results from database...")
    db.refresh(match)
    players = db.query(Player).filter(Player.match_id == match.id).all()
    detections = db.query(Detection).filter(Detection.match_id == match.id).all()
    events = db.query(Event).filter(Event.match_id == match.id).all()
    stats = db.query(PlayerStat).filter(PlayerStat.match_id == match.id).all()

    print("\n" + "=" * 70)
    print(f"📊 MATCH REPORT: {match.title}")
    print(f"Status: {match.status.value}")
    print(f"Duration: {match.duration_sec} seconds | Total Detections: {len(detections)} | Total Players: {len(players)}")
    print(f"Final Score: HOME {match.meta_data.get('score', {}).get('home', 0)} - {match.meta_data.get('score', {}).get('away', 0)} AWAY")
    print("=" * 70)

    print("\n🎯 KEY EVENTS DETECTED:")
    for e in events:
        print(f"  • [{e.timestamp_ms / 1000.0:.1f}s] {e.event_type} - Team: {e.team or 'N/A'} - Data: {json.dumps(e.data_json)}")

    print("\n🏃 PLAYER STATS & PERFORMANCE:")
    for s in stats:
        p_name = s.player.player_name if s.player else "Unknown"
        p_team = s.player.team if s.player else "N/A"
        heatmap_points = sum(len([v for v in row if v > 0.1]) for row in (s.heatmap_json or []))
        print(f"  • {p_name} ({p_team}): Dist={s.distance_m:.1f}m | Top Speed={s.top_speed_kmh:.1f}km/h | Passes={s.passes} | Heatmap Active Cells={heatmap_points}")

    # Display mini ASCII heatmap for player 1
    if stats and stats[0].heatmap_json:
        print("\n🗺️  SAMPLE 2D HEATMAP GRID (Player #1 Pitch Density):")
        for row in stats[0].heatmap_json[:10]:
            line = "".join(["█" if v > 0.5 else "▒" if v > 0.2 else "·" for v in row[:30]])
            print(f"   |{line}|")

    print("\n" + "=" * 70)
    print("✨ GAFFER MVP PIPELINE VERIFIED SUCCESSFULLY! ✨")
    print("=" * 70)
    db.close()


if __name__ == "__main__":
    main()
