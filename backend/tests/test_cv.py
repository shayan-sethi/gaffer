import cv2
import numpy as np
import pytest
from app.cv.detector import YOLOv8Detector, DetectionResult
from app.cv.tracker import CentroidTracker, TrackedObject
from app.cv.events import EventDetector
from app.cv.stats import StatsCalculator
from app.cv.calibration import PitchCalibration
from app.tasks.events import _detect_corners


def test_yolo_detector_real_empty_frame_stays_empty():
    # On real footage, a frame with no actual COCO objects (blank canvas, intro card,
    # nobody in view) must return no detections - never fabricated player positions.
    detector = YOLOv8Detector(model_path="yolov8n.pt", conf_threshold=0.3)
    frame = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
    detections = detector.detect_frame(frame)
    assert detections == []


def test_yolo_detector_synthetic_fallback_opt_in():
    # The synthetic simulation is only for the drawn-shape demo video, and only runs
    # when explicitly enabled.
    detector = YOLOv8Detector(model_path="yolov8n.pt", conf_threshold=0.3, allow_fallback_on_empty=True)
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


def test_centroid_tracker_isolated_track_does_not_poison_nearby_matches():
    # Regression test for a real bug found on production footage: Hungarian
    # assignment on a raw (ungated) distance cost matrix can force an isolated
    # track with no plausible candidate to "steal" whatever detection is least
    # bad for it, and because Hungarian minimizes the GLOBAL cost sum, that
    # cascades into two completely unrelated, genuinely-adjacent tracks losing
    # their own excellent (sub-25px) matches to the same phantom competition -
    # so all three get rejected as unmatched and re-spawned as brand new IDs,
    # instead of just the one that actually had no match. Numbers below are a
    # simplified, exact reproduction of the real failure (see tracker.py).
    tracker = CentroidTracker(max_lost_frames=30, max_distance_px=90.0)

    dets_f1 = [
        DetectionResult("player", [650, 350, 690, 410], 0.9, 0),  # "A"
        DetectionResult("player", [355, 295, 395, 355], 0.9, 0),  # "B"
        DetectionResult("player", [30, 340, 70, 400], 0.9, 0),    # "C" - isolated
    ]
    active_1 = tracker.update(dets_f1, frame_no=1)
    assert len(active_1) == 3
    by_centroid = {t.centroid: t.track_id for t in active_1}
    id_a = by_centroid[(670.0, 410.0)]
    id_b = by_centroid[(375.0, 355.0)]

    # Next frame: A and B move a small, plausible amount. C's detection is
    # genuinely missing this frame (occluded/missed) - the third detection
    # here is unrelated and far from all three tracks, not a match for anyone.
    dets_f2 = [
        DetectionResult("player", [635, 345, 675, 405], 0.9, 0),  # near A
        DetectionResult("player", [330, 300, 370, 360], 0.9, 0),  # near B
        DetectionResult("player", [810, 260, 850, 320], 0.9, 0),  # far from everyone
    ]
    active_2 = tracker.update(dets_f2, frame_no=2)
    active_ids_2 = {t.track_id for t in active_2}

    # A and B had an unambiguous, well-within-threshold match this frame and
    # must keep their original IDs - C being unmatchable is not their problem.
    assert id_a in active_ids_2
    assert id_b in active_ids_2
    # Exactly one new track spawned (for the leftover far detection) - not
    # three, which is what the bug produced.
    assert len(tracker.tracks) == 4


def test_centroid_tracker_unconfirmed_track_is_not_kept_as_reid_bait():
    # Regression test for a real bug found on production footage: a single-
    # frame detector false positive (a bright line/reflection misread as a
    # "player") spawned a track that never matched again, yet it still sat in
    # the re-ID gallery for hundreds of frames - long enough to later be
    # "found" and merged with a completely unrelated real player's trajectory,
    # corrupting that player's whole subsequent stats. A track that never gets
    # a second real sighting before expiring must be dropped outright, not
    # preserved as a re-ID candidate.
    tracker = CentroidTracker(max_lost_frames=5, max_distance_px=50.0, reid_gallery_max_age=1000)
    # A uniform-color synthetic frame - every crop produces the same jersey-
    # color histogram, so if the ghost track *were* still in the gallery, a
    # later detection would trivially "match" it by appearance (distance ~0).
    frame = np.full((400, 400, 3), (60, 60, 60), dtype=np.uint8)

    ghost_det = [DetectionResult("player", [50, 50, 90, 150], 0.9, 0)]
    active = tracker.update(ghost_det, frame_no=1, frame=frame)
    ghost_id = active[0].track_id
    assert tracker.tracks[ghost_id].hits == 1

    # Never seen again - goes lost every frame until it expires.
    for f in range(2, 10):
        tracker.update([], frame_no=f, frame=frame)
    assert ghost_id not in tracker.tracks
    # The real fix: it must not have been preserved for future re-ID either.
    assert ghost_id not in tracker.lost_gallery

    # A detection appears elsewhere nearby (well within re-ID distance budget)
    # with an identical appearance signature - if the ghost were still in the
    # gallery, this would incorrectly revive/merge into it.
    new_det = [DetectionResult("player", [150, 150, 190, 250], 0.9, 0)]
    active2 = tracker.update(new_det, frame_no=15, frame=frame)

    assert len(active2) == 1
    assert active2[0].track_id != ghost_id
    assert active2[0].hits == 1


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


def _history_from_deltas(start_frame, start_pos, deltas):
    # deltas: list of (dx_px, dy_px) applied one per frame step.
    history = [(start_frame, start_pos)]
    frame_no, (x, y) = start_frame, start_pos
    for dx, dy in deltas:
        frame_no += 1
        x, y = x + dx, y + dy
        history.append((frame_no, (x, y)))
    return history


def test_stats_calculator_single_point_history_returns_zeroed_stats():
    calc = StatsCalculator(pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0)
    player = TrackedObject(1, "player", [100, 100, 140, 180], (120, 180), 1)

    metrics = calc.compute_player_stats(player, events=[], total_frames=1)

    assert metrics["distance_m"] == 0.0
    assert metrics["top_speed_kmh"] == 0.0
    assert metrics["avg_speed_kmh"] == 0.0
    assert metrics["sprints"] == 0


def test_stats_calculator_avg_speed_matches_top_speed_at_constant_pace():
    calc = StatsCalculator(pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0)
    player = TrackedObject(1, "player", [100, 100, 140, 180], (100.0, 200.0), 1)
    # 2px/frame at 30fps, px_to_m_x = 40/1280 -> a steady ~6.75 km/h jog.
    player.history = _history_from_deltas(1, (100.0, 200.0), [(2.0, 0.0)] * 20)

    metrics = calc.compute_player_stats(player, events=[], total_frames=21)

    assert metrics["top_speed_kmh"] == metrics["avg_speed_kmh"]
    assert metrics["top_speed_kmh"] == pytest.approx(6.75)
    assert metrics["sprints"] == 0


def test_stats_calculator_counts_distinct_sustained_sprints():
    calc = StatsCalculator(pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0)
    player = TrackedObject(1, "player", [100, 100, 140, 180], (100.0, 200.0), 1)

    # Jog (~6.75 km/h, below the 20 km/h sprint-enter threshold) ...
    jog = [(2.0, 0.0)] * 5
    # ... then a sustained fast burst (~27 km/h, well over 0.8s) - one sprint ...
    burst = [(8.0, 0.0)] * 26
    # ... back to a jog (drops below the 15 km/h exit threshold, ending the sprint) ...
    # ... then a second sustained burst - a second, distinct sprint.
    player.history = _history_from_deltas(1, (100.0, 200.0), jog + burst + jog + burst)

    metrics = calc.compute_player_stats(player, events=[], total_frames=len(player.history))

    assert metrics["sprints"] == 2
    assert metrics["top_speed_kmh"] == pytest.approx(27.0)
    # Weighted average of the jog and burst speeds - strictly between the two.
    assert 6.75 < metrics["avg_speed_kmh"] < 27.0


def test_stats_calculator_sprint_below_min_duration_does_not_count():
    calc = StatsCalculator(pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0)
    player = TrackedObject(1, "player", [100, 100, 140, 180], (100.0, 200.0), 1)

    # Fast burst (~27 km/h) for only 5 frames (~0.167s) - under the 0.8s minimum
    # sustained duration, so it must not register as a sprint.
    jog = [(2.0, 0.0)] * 5
    short_burst = [(8.0, 0.0)] * 5
    player.history = _history_from_deltas(1, (100.0, 200.0), jog + short_burst + jog)

    metrics = calc.compute_player_stats(player, events=[], total_frames=len(player.history))

    assert metrics["sprints"] == 0


def test_stats_calculator_teleport_artifact_excluded_from_speed_stats():
    calc = StatsCalculator(pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0)
    player = TrackedObject(1, "player", [100, 100, 140, 180], (100.0, 200.0), 1)

    # A single-frame 800px jump (~360 km/h, a detection/Re-ID glitch) sandwiched
    # between realistic jogging speed - must be filtered out of top/avg speed,
    # not treated as the player's real top speed.
    jog = [(2.0, 0.0)] * 5
    teleport = [(800.0, 0.0)]
    player.history = _history_from_deltas(1, (100.0, 200.0), jog + teleport + jog)

    metrics = calc.compute_player_stats(player, events=[], total_frames=len(player.history))

    assert metrics["top_speed_kmh"] < 35.0


def test_compute_xg_decreases_with_distance_and_angle():
    detector = EventDetector(frame_width=1280, frame_height=720, fps=30.0)

    close_central = detector._compute_xg(distance_m=5.0, angle_deg=0.0)
    far_central = detector._compute_xg(distance_m=25.0, angle_deg=0.0)
    close_wide = detector._compute_xg(distance_m=5.0, angle_deg=40.0)

    assert 0.0 < far_central < close_central < 1.0
    assert close_wide < close_central


def test_shot_event_carries_a_plausible_xg():
    detector = EventDetector(frame_width=1280, frame_height=720, fps=30.0)

    home_player = TrackedObject(1, "player", [980, 340, 1020, 380], (1000.0, 360.0), 1)
    home_player.team = "HOME"
    ball_at_player = TrackedObject(99, "ball", [990, 350, 1010, 370], (1000.0, 360.0), 1)
    # Establishes HOME as the last possessor before the ball is struck.
    detector.process_frame([home_player, ball_at_player], frame_no=1)

    # 10 frames later (0.33s, inside the shot-velocity window) the ball is deep
    # in the box dead in front of goal, moving fast and straight at it.
    ball_struck = TrackedObject(99, "ball", [1190, 350, 1210, 370], (1200.0, 360.0), 11)
    events = detector.process_frame([home_player, ball_struck], frame_no=11)

    shot_evts = [e for e in events if e.event_type == "SHOT"]
    assert len(shot_evts) == 1
    assert shot_evts[0].team == "HOME"
    xg = shot_evts[0].data_json["xg"]
    # Point-blank, dead-central shot - xG should be high, not a token nonzero value.
    assert 0.7 < xg <= 1.0


def test_zone_touches_attributed_by_each_teams_attacking_direction():
    # HOME attacks the RIGHT (high-x) goal, AWAY attacks the LEFT (low-x) goal -
    # so the same raw ball x-position is a different zone for each team.
    detector = EventDetector(frame_width=1200, frame_height=600, fps=30.0, pitch_bounds=(0, 0, 1200, 600))

    def touch(track_id, team, x, frame_no):
        player = TrackedObject(track_id, "player", [x - 20, 530, x + 20, 570], (float(x), 550.0), frame_no)
        player.team = team
        ball = TrackedObject(99, "ball", [x - 10, 545, x + 10, 555], (float(x), 550.0), frame_no)
        detector.process_frame([player, ball], frame_no=frame_no)

    touch(1, "HOME", 100, 1)    # HOME's own defensive third
    touch(1, "HOME", 600, 101)  # midfield
    touch(1, "HOME", 1000, 201)  # HOME's attacking third
    touch(2, "AWAY", 1000, 301)  # AWAY's own defensive third (mirrored)
    touch(2, "AWAY", 100, 401)   # AWAY's attacking third (mirrored)

    assert detector.zone_touches["HOME"] == {"DEFENSIVE": 1, "MIDDLE": 1, "ATTACKING": 1}
    assert detector.zone_touches["AWAY"] == {"DEFENSIVE": 1, "MIDDLE": 0, "ATTACKING": 1}


def test_detect_corners_attributes_team_by_pitch_side():
    pitch_bounds = (0, 0, 1200, 600)

    # Ball goes undetected for a real stoppage (gap 190 >> max_gap_ball 15) then
    # reappears pinned in the top-right corner - HOME attacks right, so it's HOME's.
    events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (600.0, 300.0), 200: (1150.0, 50.0)},
        pitch_bounds=pitch_bounds,
        frame_width=1200,
        frame_height=600,
        fps=30.0,
        max_gap_ball=15,
    )

    assert len(events) == 1
    assert events[0].event_type == "CORNER"
    assert events[0].team == "HOME"
    assert events[0].frame_no == 200


def test_detect_corners_ignores_stoppages_away_from_a_corner():
    events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (600.0, 300.0), 200: (600.0, 300.0)},  # reappears mid-pitch
        pitch_bounds=(0, 0, 1200, 600),
        frame_width=1200,
        frame_height=600,
        fps=30.0,
        max_gap_ball=15,
    )

    assert events == []


def test_detect_corners_ignores_ordinary_sampling_gaps():
    events = _detect_corners(
        ball_sorted_frames=[10, 20],  # gap of 10, under max_gap_ball
        ball_positions={10: (600.0, 300.0), 20: (1150.0, 50.0)},
        pitch_bounds=(0, 0, 1200, 600),
        frame_width=1200,
        frame_height=600,
        fps=30.0,
        max_gap_ball=15,
    )

    assert events == []


# A trapezoid modeling a real fixed, angled camera (same geometry as
# test_calibration.py): the near touchline (large y) spans far more pixels
# than the far touchline (small y) for the same 40m real-world length.
_CALIB_TL, _CALIB_TR, _CALIB_BR, _CALIB_BL = (400.0, 100.0), (880.0, 100.0), (1230.0, 650.0), (50.0, 650.0)


def _far_touchline_calibration():
    return PitchCalibration([_CALIB_TL, _CALIB_TR, _CALIB_BR, _CALIB_BL], pitch_length_m=40.0, pitch_width_m=20.0)


def test_stats_calculator_calibration_fixes_the_linear_approximations_skew():
    player = TrackedObject(1, "player", [0, 0, 0, 0], _CALIB_TL, 1)
    # A run entirely along the far (narrow-in-pixels) touchline, between the
    # two far corners - a real, known ~480px move worth exactly 40m.
    player.history = [(1, _CALIB_TL), (2, _CALIB_TR)]

    calibrated = StatsCalculator(
        pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720,
        fps=30.0, calibration=_far_touchline_calibration(),
    )
    # The old behavior: a flat ratio derived from the pitch's overall bounding
    # box (as tasks/stats.py builds it from the pitch mask), which - because
    # the near touchline is wider - understates real distance for play that
    # happens up near the far touchline.
    uncalibrated = StatsCalculator(
        pitch_length_m=40.0, pitch_width_m=20.0, frame_width=1280, frame_height=720, fps=30.0,
        pitch_px_width=_CALIB_BR[0] - _CALIB_BL[0], pitch_px_height=_CALIB_BL[1] - _CALIB_TL[1],
    )

    calibrated_metrics = calibrated.compute_player_stats(player, events=[], total_frames=2)
    uncalibrated_metrics = uncalibrated.compute_player_stats(player, events=[], total_frames=2)

    assert calibrated_metrics["distance_m"] == pytest.approx(40.0, abs=0.5)
    # The uncalibrated figure is off by a wide margin - not a rounding difference.
    assert uncalibrated_metrics["distance_m"] < 30.0


def test_event_detector_to_world_m_uses_calibration_when_set():
    calibration = _far_touchline_calibration()
    detector = EventDetector(frame_width=1280, frame_height=720, fps=30.0, calibration=calibration)

    assert detector._to_world_m(*_CALIB_TR) == pytest.approx(calibration.pixel_to_world_m(*_CALIB_TR), abs=1e-6)
    # And it must actually be routing through calibration, not silently
    # falling back to the linear approximation alongside it.
    fallback_x = (_CALIB_TR[0] - detector.pitch_x1) * detector.px_to_m_x
    assert detector._to_world_m(*_CALIB_TR)[0] != pytest.approx(fallback_x, abs=0.5)


def _forward_project_far_touchline(x_m, y_m):
    world = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]], dtype=np.float32)
    image = np.array([_CALIB_TL, _CALIB_TR, _CALIB_BR, _CALIB_BL], dtype=np.float32)
    h = cv2.getPerspectiveTransform(world, image)
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    px = cv2.perspectiveTransform(pt, h)
    return float(px[0, 0, 0]), float(px[0, 0, 1])


def test_detect_corners_with_calibration_attributes_by_real_world_proximity():
    calibration = _far_touchline_calibration()
    # 2.7m from the AWAY (0,0) corner in real terms - clearly "a corner", even
    # though it sits at the far/narrow-in-pixels end of the pitch.
    reappear_px = _forward_project_far_touchline(2.0, 1.8)

    events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (700.0, 350.0), 200: reappear_px},
        pitch_bounds=None,
        frame_width=1280, frame_height=720, fps=30.0, max_gap_ball=15,
        calibration=calibration,
    )

    assert len(events) == 1
    assert events[0].team == "AWAY"


def test_detect_corners_with_calibration_ignores_points_far_from_any_corner():
    calibration = _far_touchline_calibration()
    midfield_px = _forward_project_far_touchline(20.0, 10.0)

    events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (700.0, 350.0), 200: midfield_px},
        pitch_bounds=None,
        frame_width=1280, frame_height=720, fps=30.0, max_gap_ball=15,
        calibration=calibration,
    )

    assert events == []


def test_detect_corners_calibration_catches_a_far_corner_the_pixel_fraction_heuristic_misses():
    calibration = _far_touchline_calibration()
    # 1.4m from the far AWAY corner (0,0) - unambiguously "at the corner" in
    # real-world terms.
    reappear_px = _forward_project_far_touchline(1.0, 1.0)

    calibrated_events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (700.0, 350.0), 200: reappear_px},
        pitch_bounds=None, frame_width=1280, frame_height=720, fps=30.0, max_gap_ball=15,
        calibration=calibration,
    )

    # The pixel-fraction fallback, using the trapezoid's own bounding box: an
    # 8% margin of the *whole* bbox width barely reaches into the narrow far
    # end, so this same real-world corner point isn't "near the edge" in
    # pixel terms there - exactly the perspective distortion calibration fixes.
    pitch_bounds = (
        int(min(_CALIB_TL[0], _CALIB_BL[0])), int(min(_CALIB_TL[1], _CALIB_TR[1])),
        int(max(_CALIB_TR[0], _CALIB_BR[0])), int(max(_CALIB_BL[1], _CALIB_BR[1])),
    )
    uncalibrated_events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (700.0, 350.0), 200: reappear_px},
        pitch_bounds=pitch_bounds, frame_width=1280, frame_height=720, fps=30.0, max_gap_ball=15,
        calibration=None,
    )

    assert len(calibrated_events) == 1
    assert calibrated_events[0].team == "AWAY"
    assert uncalibrated_events == []
