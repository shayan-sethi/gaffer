import numpy as np
from app.cv.detector import YOLOv8Detector, DetectionResult
from app.cv.tracker import CentroidTracker, TrackedObject
from app.cv.events import EventDetector
from app.cv.stats import StatsCalculator


def test_yolo_detector_and_fallback():
    detector = YOLOv8Detector(model_path="yolov8n.pt", conf_threshold=0.3)
    frame = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
    detections = detector.detect_frame(frame)

    assert len(detections) > 0
    player_dets = [d for d in detections if d.class_name == "player"]
    ball_dets = [d for d in detections if d.class_name == "ball"]

    assert len(player_dets) == 10
    assert len(ball_dets) == 1
    assert all(len(d.bbox) == 4 for d in detections)


def test_centroid_tracker():
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)

    # Frame 1
    dets_f1 = [
        DetectionResult("player", [100, 200, 140, 280], 0.9, 0),
        DetectionResult("player", [300, 200, 340, 280], 0.9, 0),
        DetectionResult("ball", [200, 250, 220, 270], 0.95, 32),
    ]
    active_1 = tracker.update(dets_f1, frame_no=1)
    assert len(active_1) == 3
    track_ids_1 = {t.track_id for t in active_1}

    # Frame 2: slightly moved objects
    dets_f2 = [
        DetectionResult("player", [105, 202, 145, 282], 0.9, 0),
        DetectionResult("player", [305, 202, 345, 282], 0.9, 0),
        DetectionResult("ball", [210, 255, 230, 275], 0.95, 32),
    ]
    active_2 = tracker.update(dets_f2, frame_no=2)
    assert len(active_2) == 3
    track_ids_2 = {t.track_id for t in active_2}

    # Verify ID persistence across frames
    assert track_ids_1 == track_ids_2


def test_event_detector_and_goals():
    event_detector = EventDetector(frame_width=1280, frame_height=720, fps=30.0)

    # Simulate player 1 passing to player 2
    p1 = TrackedObject(1, "player", [100, 200, 140, 280], (120, 280), 1)
    p1.team = "HOME"
    p2 = TrackedObject(2, "player", [250, 200, 290, 280], (270, 280), 1)
    p2.team = "HOME"

    ball_at_p1 = TrackedObject(99, "ball", [120, 270, 140, 290], (130, 280), 1)

    # Frame 1: ball near p1
    evts_1 = event_detector.process_frame([p1, p2, ball_at_p1], frame_no=1)

    # Frame 10: ball near p2 (completed pass)
    ball_at_p2 = TrackedObject(99, "ball", [260, 270, 280, 290], (270, 280), 10)
    evts_2 = event_detector.process_frame([p1, p2, ball_at_p2], frame_no=10)

    pass_evts = [e for e in evts_2 if e.event_type == "PASS"]
    assert len(pass_evts) == 1
    assert pass_evts[0].player_track_id == 1


def test_stats_calculator_heatmaps_and_distance():
    calc = StatsCalculator(pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0)

    player = TrackedObject(1, "player", [100, 100, 140, 180], (120, 180), 1)
    # Simulate movement across 30 frames
    for f in range(2, 31):
        player.history.append((f, (120.0 + f * 5.0, 180.0 + f * 2.0)))

    metrics = calc.compute_player_stats(player, events=[], total_frames=30)

    assert metrics["distance_m"] > 0.0
    assert metrics["top_speed_kmh"] > 0.0
    assert len(metrics["heatmap_json"]) == 20  # 20 rows
    assert len(metrics["heatmap_json"][0]) == 30  # 30 cols
