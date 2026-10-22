from typing import Tuple

import cv2
import numpy as np
import pytest
import torch
from app.cv.detector import YOLOv8Detector, DetectionResult
from app.cv.tracker import CentroidTracker, TrackedObject
from app.cv.bot_sort_tracker import BotSortPlayerTracker
from app.cv.events import EventDetector
from app.cv.stats import StatsCalculator
from app.cv.calibration import PitchCalibration
from app.tasks.events import _detect_corners, _detect_throw_ins


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


def test_dedup_overlapping_player_detections_drops_nested_spurious_box():
    # Regression test for a real bug found on production footage: an awkward pose
    # (arm raised mid-reach) made the model output a second, smaller, lower-
    # confidence box nested almost entirely inside the real full-body one - just
    # the arm/shoulder read as its own "player". IoU is small (different-sized
    # boxes), so the model's own NMS didn't merge them, and the tracker matched
    # both to separate tracks with conflicting team labels. Real bboxes/confidences
    # from that footage (see detector.py's PLAYER_CONTAINMENT_DEDUP_THRESHOLD).
    detector = YOLOv8Detector.__new__(YOLOv8Detector)  # skip model loading - pure post-processing logic
    full_body = DetectionResult("player", [520.29, 448.32, 659.92, 660.15], 0.629, 1)
    spurious_arm = DetectionResult("player", [579.83, 452.17, 663.56, 500.90], 0.409, 1)

    kept = detector._dedup_overlapping_player_detections([full_body, spurious_arm])

    assert len(kept) == 1
    assert kept[0] is full_body


def test_dedup_overlapping_player_detections_keeps_genuinely_separate_players():
    # Two real, adjacent players (e.g. one marking the other) must both survive -
    # dedup should only fire when one box is almost entirely nested in the other,
    # not for ordinary partial overlap between two distinct people.
    detector = YOLOv8Detector.__new__(YOLOv8Detector)
    player_a = DetectionResult("player", [100, 200, 180, 400], 0.9, 1)
    player_b = DetectionResult("player", [160, 210, 240, 410], 0.85, 1)

    kept = detector._dedup_overlapping_player_detections([player_a, player_b])

    assert len(kept) == 2


class _FakeBoxes:
    def __init__(self, ids, clss, confs, xyxys):
        self.id = torch.tensor(ids) if ids is not None else None
        self.cls = torch.tensor(clss)
        self.conf = torch.tensor(confs, dtype=torch.float32)
        self.xyxy = torch.tensor(xyxys, dtype=torch.float32)


class _FakeYoloResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeTrackModel:
    """Stands in for ultralytics' YOLO model - returns scripted per-call results
    instead of running real inference, so BotSortPlayerTracker's box-to-TrackedObject
    conversion logic can be tested without loading model weights."""

    def __init__(self, frame_results):
        self.frame_results = list(frame_results)
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self.frame_results.pop(0)]


_CLASS_MAP = {0: "ball", 1: "player"}


def _make_bot_sort_tracker(model) -> BotSortPlayerTracker:
    """BotSortPlayerTracker.__init__ loads its own real model from a path (see its
    docstring for why: a shared model object's tracker state gets corrupted by
    interleaved plain-inference calls) - tests skip that and inject a fake instead,
    the same __new__-bypass pattern used for YOLOv8Detector elsewhere in this file."""
    tracker = BotSortPlayerTracker.__new__(BotSortPlayerTracker)
    tracker.model = model
    tracker.class_map = _CLASS_MAP
    tracker.tracker_yaml = "botsort.yaml"
    tracker.conf_threshold = 0.3
    tracker.iou_threshold = 0.45
    tracker.img_size = 640
    tracker.device = "cpu"
    tracker.tracks = {}
    tracker._team_reference_centroids = None
    tracker._team_assign_counter = 0
    return tracker


def test_bot_sort_player_tracker_converts_boxes_and_separates_ball():
    # A ball detection must never become a player TrackedObject (BoT-SORT's own
    # ID for it is discarded - see class docstring for why the ball needs its
    # own single-object tracker rather than a general multi-object one).
    model = _FakeTrackModel([
        _FakeYoloResult(_FakeBoxes(
            ids=[5, 9], clss=[1, 0], confs=[0.9, 0.5],
            xyxys=[[100, 100, 140, 200], [300, 300, 310, 310]],
        )),
    ])
    tracker = _make_bot_sort_tracker(model)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    active, player_dets, ball_dets = tracker.update(frame, frame_no=1)

    assert len(active) == 1
    assert active[0].track_id == 5
    assert active[0].class_name == "player"
    assert active[0].centroid == (120.0, 200.0)  # bottom-center
    assert len(player_dets) == 1
    assert player_dets[0].class_name == "player"
    assert len(ball_dets) == 1
    assert ball_dets[0].class_name == "ball"
    assert 9 not in tracker.tracks


def test_bot_sort_player_tracker_keeps_same_id_across_frames():
    # The whole point of swapping in BoT-SORT: a player who's briefly re-detected
    # under the same ID must extend the same TrackedObject (growing .history),
    # not spawn a new one.
    model = _FakeTrackModel([
        _FakeYoloResult(_FakeBoxes(ids=[5], clss=[1], confs=[0.9], xyxys=[[100, 100, 140, 200]])),
        _FakeYoloResult(_FakeBoxes(ids=[5], clss=[1], confs=[0.9], xyxys=[[110, 100, 150, 200]])),
    ])
    tracker = _make_bot_sort_tracker(model)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    active_1, _, _ = tracker.update(frame, frame_no=1)
    active_2, _, _ = tracker.update(frame, frame_no=2)

    assert active_1[0] is active_2[0]
    assert len(tracker.tracks) == 1
    assert tracker.tracks[5].history == [(1, (120.0, 200.0)), (2, (130.0, 200.0))]


def test_bot_sort_player_tracker_prune_drops_single_hit_ghosts_only():
    # Regression test for the real over-counting bug: BoT-SORT's default
    # new_track_thresh confidently mints a track ID for a single-frame false
    # positive that's never seen again. Pruning must drop only those (hits==1),
    # not a genuinely-confirmed track that's merely short-lived (hits>=2).
    model = _FakeTrackModel([
        _FakeYoloResult(_FakeBoxes(
            ids=[1, 2], clss=[1, 1], confs=[0.9, 0.9],
            xyxys=[[100, 100, 140, 200], [300, 300, 340, 400]],
        )),
        _FakeYoloResult(_FakeBoxes(ids=[1], clss=[1], confs=[0.9], xyxys=[[105, 100, 145, 200]])),
    ])
    tracker = _make_bot_sort_tracker(model)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)

    tracker.update(frame, frame_no=1)  # spawns tracks 1 and 2
    tracker.update(frame, frame_no=2)  # only track 1 matched again; 2 never reappears

    assert tracker.tracks[1].hits == 2
    assert tracker.tracks[2].hits == 1

    tracker.prune_unconfirmed_tracks()

    assert 1 in tracker.tracks
    assert 2 not in tracker.tracks


def test_bot_sort_player_tracker_drops_players_outside_pitch_mask():
    # Sideline spectators/bystanders get detected as "player" just as readily as
    # real players - pitch_mask filtering must still apply, same as the
    # CentroidTracker path in track.py.
    model = _FakeTrackModel([
        _FakeYoloResult(_FakeBoxes(
            ids=[1, 2], clss=[1, 1], confs=[0.9, 0.9],
            xyxys=[[100, 100, 140, 200], [300, 300, 340, 400]],
        )),
    ])
    tracker = _make_bot_sort_tracker(model)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)

    class OnlyLeftHalf:
        def contains_point(self, x, y):
            return x < 250

    active, _, _ = tracker.update(frame, frame_no=1, pitch_mask=OnlyLeftHalf())

    assert len(active) == 1
    assert active[0].track_id == 1
    assert 2 not in tracker.tracks


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


def test_centroid_tracker_ball_rejects_static_decoy_during_loss_recovery():
    # Regression test for a real bug found on production footage: a static
    # background object (a sponsor board graphic) got confidently misdetected
    # as "ball" at the exact same pixel spot for 2.5+ seconds, and once the
    # real ball went briefly undetected, the ball track's own loss-recovery
    # jump budget eventually widened enough to accept that decoy - hijacking
    # the track and causing a missed GOAL plus a wildly wrong xG estimate for
    # the shot that immediately followed. A real ball in flight cannot recur
    # at a near-identical position over and over; a static decoy can and does,
    # and that recurrence is exactly the signal _is_static_ball_candidate uses
    # to reject it, however plausible the decoy looks by distance alone.
    tracker = CentroidTracker(max_lost_frames=30, max_distance_px=100.0)

    real_ball_det = DetectionResult("ball", [95, 95, 105, 105], 0.9, 32)  # center (100, 100)
    active_1 = tracker.update([real_ball_det], frame_no=1)
    assert len(active_1) == 1
    ball_id = active_1[0].track_id

    decoy_det = DetectionResult("ball", [595, 95, 605, 105], 0.95, 32)  # center (600, 100), far away
    # Feed the decoy at the same static spot across several frames while the
    # real ball goes undetected. By frame 5, max_jump = min(700, 140*5) = 700
    # comfortably covers the 500px gap to the decoy - without the static-
    # candidate check, this is exactly the frame the old bug would hijack the
    # track onto the decoy. The fix must keep rejecting it anyway, since by
    # then it's been logged well past STATIC_BALL_CANDIDATE_MIN_SIGHTINGS.
    for frame_no in range(2, 6):
        active = tracker.update([decoy_det], frame_no=frame_no)
        assert active == []
        assert tracker.tracks[ball_id].centroid == (100.0, 100.0)

    # The real ball reappears near its last known position - still accepted
    # normally, since a couple of real sightings there don't cross the
    # static-candidate threshold the way the repeated decoy did.
    real_ball_again = DetectionResult("ball", [100, 97, 110, 107], 0.9, 32)  # center (105, 102)
    active_6 = tracker.update([real_ball_again], frame_no=6)
    assert len(active_6) == 1
    assert active_6[0].track_id == ball_id
    assert tracker.tracks[ball_id].centroid == (105.0, 102.0)


def test_centroid_tracker_ball_drops_lock_pinned_to_a_walking_player():
    # Regression test for a real bug found on production footage (Barcelona vs
    # Espanyol tactical-cam clip): a "ball" track stayed locked onto a slowly-
    # walking match official for 10+ continuous seconds. _is_static_ball_
    # candidate can't catch this - a moving decoy never trips it, and since the
    # false candidate keeps winning ordinary trajectory-continuity matching,
    # the track is never "lost" long enough for that check to even run (it
    # only runs during *recovery* from a loss - see _update_ball). Meanwhile a
    # real ball sitting elsewhere gets excluded every frame by the max_jump
    # gate, since predicted position keeps following the wrong (player's)
    # trajectory instead.
    tracker = CentroidTracker(max_lost_frames=30, max_distance_px=100.0)

    ball_det_1 = DetectionResult("ball", [95, 95, 105, 105], 0.95, 32)  # center (100, 100)
    active_1 = tracker.update([ball_det_1], frame_no=1)
    assert len(active_1) == 1
    original_ball_id = active_1[0].track_id

    # Frames 2-51: only a "ball" detection pinned inside a slowly-walking
    # player's own box is available - simulates a shoe/badge misdetection
    # that drifts exactly however that one person walks (2px/frame, well
    # under max_jump=140 every frame, so it's never "lost").
    for frame_no in range(2, 52):
        false_x = 100 + (frame_no - 2) * 2
        false_det = DetectionResult("ball", [false_x - 5, 95, false_x + 5, 105], 0.95, 32)
        player_det = DetectionResult("player", [false_x - 30, 70, false_x + 30, 130], 0.9, 0)
        active = tracker.update([false_det, player_det], frame_no=frame_no)
        ball_tracks = [t for t in active if t.class_name == "ball"]
        assert len(ball_tracks) == 1  # still (wrongly) tracking the pinned decoy
        assert ball_tracks[0].track_id == original_ball_id

    # Frame 52 (the 51st consecutive pinned frame, past BALL_PINNED_TO_PLAYER_
    # MAX_FRAMES=50): a real ball newly appears far away, at a position that
    # doesn't touch the player and hasn't been logged before (so it isn't
    # mistaken for a *static* decoy either) - but is still far outside the
    # pinned track's max_jump budget, so only dropping the false lock first
    # lets it be picked up at all.
    false_x = 100 + (52 - 2) * 2
    false_det = DetectionResult("ball", [false_x - 5, 95, false_x + 5, 105], 0.95, 32)
    player_det = DetectionResult("player", [false_x - 30, 70, false_x + 30, 130], 0.9, 0)
    real_ball_det = DetectionResult("ball", [695, 95, 705, 105], 0.5, 32)  # center (700, 100)

    active_52 = tracker.update([false_det, player_det, real_ball_det], frame_no=52)
    ball_tracks_52 = [t for t in active_52 if t.class_name == "ball"]

    assert original_ball_id not in tracker.tracks
    assert len(ball_tracks_52) == 1
    assert ball_tracks_52[0].track_id != original_ball_id
    assert ball_tracks_52[0].centroid == (700.0, 100.0)


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


def test_centroid_tracker_rejects_a_closer_but_wrong_colored_match():
    # Regression test for a real bug found on production footage: live frame-
    # to-frame matching (_match_detections) gated purely on position distance,
    # with no color/appearance check at all - only a track that had been LOST
    # for a while got its color checked, via the separate re-ID path. That
    # meant a real player's track could silently latch onto a completely
    # different, merely position-adjacent person (a linesman, a camera
    # operator, bench staff) any time that person's detection happened to be
    # closer to the track's predicted position than the real player's own
    # detection was that frame - several tracks on real footage flip-flopped
    # between different real identities this way. A color-consistency gate on
    # the match itself (not just re-ID) should reject the closer-but-wrong-
    # colored detection and keep the track on its real, correctly-colored one.
    frame = np.full((400, 400, 3), (0, 100, 0), dtype=np.uint8)  # grass-green background

    def paint_torso(img, bbox, bgr_color):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        ty1, ty2 = y1 + int(h * 0.15), y1 + int(h * 0.55)
        img[ty1:ty2, x1:x2] = bgr_color

    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)
    navy_bgr = (128, 0, 0)
    orange_bgr = (0, 140, 255)

    bbox = [100, 100, 140, 200]
    f = frame.copy()
    paint_torso(f, bbox, navy_bgr)

    # Establish enough real color history on this track for the gate to trust
    # it as a reference (see MIN_TRACK_COLOR_SAMPLES_FOR_GATE).
    track_id = None
    for i in range(12):
        active = tracker.update([DetectionResult("player", bbox, 0.9, 1)], frame_no=i + 1, frame=f)
        track_id = active[0].track_id
    assert len(tracker.tracks[track_id].color_samples) >= tracker.MIN_TRACK_COLOR_SAMPLES_FOR_GATE

    # Next frame: the real player moved a small, plausible amount (still navy).
    # A DIFFERENT, orange-kitted person is standing even closer to the track's
    # predicted position - purely on distance, Hungarian assignment would
    # prefer the orange detection.
    real_bbox = [108, 108, 148, 208]
    wrong_bbox = [104, 104, 144, 204]  # closer to the predicted position
    f2 = frame.copy()
    paint_torso(f2, real_bbox, navy_bgr)
    paint_torso(f2, wrong_bbox, orange_bgr)

    active = tracker.update(
        [DetectionResult("player", wrong_bbox, 0.9, 1), DetectionResult("player", real_bbox, 0.9, 1)],
        frame_no=13,
        frame=f2,
    )

    updated_track = tracker.tracks[track_id]
    # The track must have followed its real, navy-colored continuation - not
    # been hijacked by the closer, wrong-colored detection.
    assert updated_track.bbox == real_bbox
    assert updated_track.consecutive_lost == 0
    # The orange detection must NOT have been folded into this track - it
    # should show up as a distinct, separate track instead.
    other_track_ids = {t.track_id for t in active} - {track_id}
    assert len(other_track_ids) == 1
    other_track = tracker.tracks[next(iter(other_track_ids))]
    assert other_track.bbox == wrong_bbox


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


def test_sample_color_rejects_crops_too_dark_to_carry_real_color():
    # Regression test for a real bug found on production night-match footage: a
    # torso crop sitting outside the floodlight pools is near-black noise, not a
    # dimmer version of the true jersey color - averaging that noise into a
    # track's color history was confirmed to be the root cause of team
    # misassignment (two different jersey colors, and even the referee, ending
    # up in the same k-means cluster). A crop this dark must be rejected
    # outright rather than treated as a (misleading) color sample.
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)
    dark_frame = np.full((400, 400, 3), (8, 6, 7), dtype=np.uint8)
    assert tracker._sample_color(dark_frame, [50, 50, 90, 150]) is None


def test_sample_color_returns_hue_saturation_for_well_lit_crops():
    # A well-lit crop should be accepted and distinguish a saturated jersey
    # color (blue) from a neutral one (grey) via hue/saturation, not brightness.
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)
    blue_frame = np.full((400, 400, 3), (200, 80, 30), dtype=np.uint8)  # BGR: blue-dominant
    grey_frame = np.full((400, 400, 3), (180, 180, 180), dtype=np.uint8)  # BGR: neutral

    blue_hsv = tracker._sample_color(blue_frame, [50, 50, 90, 150])
    grey_hsv = tracker._sample_color(grey_frame, [50, 50, 90, 150])

    assert blue_hsv is not None and grey_hsv is not None
    _, blue_sat, blue_val = blue_hsv
    _, grey_sat, grey_val = grey_hsv
    assert blue_sat > grey_sat + 50  # blue jersey reads as clearly more saturated
    assert blue_val >= tracker.MIN_COLOR_SAMPLE_VALUE
    assert grey_val >= tracker.MIN_COLOR_SAMPLE_VALUE


def test_assign_teams_clusters_by_hue_saturation_not_brightness():
    # Regression test for the real clustering failure: two players in the same
    # (grey/white) jersey but under different lighting must land on the same
    # team, and must not be pulled towards a differently-colored (blue) jersey
    # just because their brightness happens to be closer to it. Hue+saturation
    # (this test) is invariant to that; the old raw-BGR-brightness-entangled
    # feature was not (see _sample_color, _assign_teams).
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)

    def make_player(track_id: int, hsv_sample: Tuple[float, float, float]) -> TrackedObject:
        p = TrackedObject(track_id, "player", [0, 0, 40, 100], (float(track_id) * 100, 100.0), 1)
        p.color_samples = [hsv_sample] * 5
        return p

    # Two "blue" players: same hue/saturation, very different brightness.
    blue_bright = make_player(1, (110.0, 160.0, 130.0))
    blue_dim = make_player(2, (108.0, 150.0, 60.0))
    # Two "grey" players: near-zero saturation, very different brightness.
    grey_bright = make_player(3, (0.0, 5.0, 200.0))
    grey_dim = make_player(4, (0.0, 8.0, 70.0))

    tracker.tracks = {p.track_id: p for p in [blue_bright, blue_dim, grey_bright, grey_dim]}
    tracker._team_assign_counter = 9  # next _assign_teams() call hits the %10==0 cluster step
    tracker._assign_teams()

    assert blue_bright.team == blue_dim.team
    assert grey_bright.team == grey_dim.team
    assert blue_bright.team != grey_bright.team


def test_assign_teams_labels_a_third_kit_color_as_referee():
    # Regression test using real (hue, saturation) values measured from actual
    # production footage: 11 confirmed real players in two jersey colors (navy,
    # white), plus one confirmed on-pitch referee in a third color (green) that
    # sits far outside both team clusters - a referee wears a kit distinct from
    # both teams by design, so this is real signal, not a guess (see
    # TeamAssignmentMixin.REFEREE_OUTLIER_DIST for the measured player-vs-referee
    # distance gap this threshold is calibrated against).
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)

    def make_player(track_id: int, hue: float, sat: float) -> TrackedObject:
        p = TrackedObject(track_id, "player", [0, 0, 40, 100], (float(track_id) * 100, 100.0), 1)
        p.color_samples = [(hue, sat, 100.0)] * 5
        return p

    real_hue_sat = [
        (93, 66), (117, 52), (113, 79), (115, 67), (79, 62), (124, 91),
        (126, 55), (112, 87), (105, 63), (114, 57), (118, 71),
    ]
    players = [make_player(i + 1, h, s) for i, (h, s) in enumerate(real_hue_sat)]
    referee = make_player(99, 68, 73)  # confirmed real referee sample

    tracker.tracks = {p.track_id: p for p in players + [referee]}
    tracker._team_assign_counter = 9  # next _assign_teams() call hits the %10==0 cluster step
    tracker._assign_teams()

    assert referee.team == "REFEREE"
    assert {p.team for p in players} <= {"HOME", "AWAY"}


def test_assign_teams_survives_multiple_non_team_kit_colors():
    # Regression test for a professional match, which (unlike the single-referee
    # case above) has SEVERAL non-team kit colors in frame at once: each team's
    # own goalkeeper (in a color distinct from their own outfield kit) plus a
    # referee/assistant linesman. Confirmed on real footage (Barcelona vs
    # Espanyol) that without refitting the team centroids after removing
    # first-pass outliers, these extra non-team colors drag both centroids far
    # enough off the real jersey colors that goalkeepers and officials get
    # split across both HOME and AWAY buckets instead of excluded as REFEREE.
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)

    def make_player(track_id: int, hue: float, sat: float) -> TrackedObject:
        p = TrackedObject(track_id, "player", [0, 0, 40, 100], (float(track_id) * 100, 100.0), 1)
        p.color_samples = [(hue, sat, 100.0)] * 5
        return p

    # HOME: navy/blue kit, tight real-world spread.
    home_hue_sat = [(113, 79), (115, 67), (124, 91), (112, 87), (114, 57), (118, 71)]
    # AWAY: yellow kit, tight real-world spread, far from HOME in hue.
    away_hue_sat = [(28, 70), (32, 66), (25, 74), (35, 62), (30, 69), (27, 73)]
    home_players = [make_player(i + 1, h, s) for i, (h, s) in enumerate(home_hue_sat)]
    away_players = [make_player(i + 7, h, s) for i, (h, s) in enumerate(away_hue_sat)]

    # Two goalkeepers, both in the same third (pink) kit color, positioned so a
    # naive single-pass fit splits them one to each nominal side instead of
    # recognizing them as one shared non-team color.
    keeper_home_side = make_player(90, 160, 45)
    keeper_away_side = make_player(91, 165, 42)
    # A linesman in hi-vis orange - yet another distinct non-team color.
    linesman = make_player(99, 15, 88)

    tracker.tracks = {
        p.track_id: p
        for p in home_players + away_players + [keeper_home_side, keeper_away_side, linesman]
    }
    tracker._team_assign_counter = 9  # next _assign_teams() call hits the %10==0 cluster step
    tracker._assign_teams()

    # Which literal label ("HOME" vs "AWAY") lands on which real cluster is
    # arbitrary (kmeans2 label order isn't meaningful) - what matters is that
    # each real team is internally consistent and distinct from the other.
    home_teams = {p.team for p in home_players}
    away_teams = {p.team for p in away_players}
    assert len(home_teams) == 1
    assert len(away_teams) == 1
    assert home_teams != away_teams
    assert home_teams <= {"HOME", "AWAY"}

    assert keeper_home_side.team == "REFEREE"
    assert keeper_away_side.team == "REFEREE"
    assert linesman.team == "REFEREE"


def test_assign_teams_recovers_from_an_early_misclassification():
    # Regression test for a real failure mode confirmed on production footage
    # (Barcelona vs Espanyol): a track ID can briefly represent the WRONG
    # physical person right after it's created (e.g. a Hungarian-assignment
    # mismatch hands it a goalkeeper's identity for its first few real
    # detections) before settling onto the real outfield player it tracks for
    # the rest of the match. Locking on that first, unrepresentative
    # classification permanently mislabeled the track as "REFEREE" even
    # though 95%+ of its life was a normal, correctly-colored player - three
    # real tracks on one clip were lost to exactly this. Requiring two
    # consecutive agreeing classifications before locking (see
    # TrackedObject._pending_team) should let this one recover instead.
    tracker = CentroidTracker(max_lost_frames=10, max_distance_px=100.0)

    def make_player(track_id: int, hue: float, sat: float) -> TrackedObject:
        p = TrackedObject(track_id, "player", [0, 0, 40, 100], (float(track_id) * 100, 100.0), 1)
        p.color_samples = [(hue, sat, 100.0)] * 5
        return p

    home_hue_sat = [(113, 79), (115, 67), (124, 91), (112, 87), (114, 57), (118, 71)]
    away_hue_sat = [(28, 70), (32, 66), (25, 74), (35, 62), (30, 69), (27, 73)]
    home_players = [make_player(i + 1, h, s) for i, (h, s) in enumerate(home_hue_sat)]
    away_players = [make_player(i + 7, h, s) for i, (h, s) in enumerate(away_hue_sat)]

    # This track starts out looking like the pink-kit goalkeeper (a real
    # outlier, same as the keeper in the test above)...
    switcher = make_player(50, 160, 45)

    tracker.tracks = {p.track_id: p for p in home_players + away_players + [switcher]}
    tracker._team_assign_counter = 9
    tracker._assign_teams()
    assert switcher.team == "REFEREE"
    assert switcher.team_locked is False  # only one agreeing read so far - not locked yet
    # Which literal label ("HOME" vs "AWAY") lands on the navy/blue cluster is
    # arbitrary (kmeans2 label order isn't meaningful) - what matters is that
    # the switcher later agrees with THIS cluster's real players, whichever
    # string that turned out to be.
    home_label = home_players[0].team
    assert home_label in {"HOME", "AWAY"}

    # ...then, a few seconds later, the SAME track ID has settled onto a real
    # player from the navy/blue side instead (its rolling color_samples window
    # now reflects a completely different physical person - see
    # CentroidTracker._appearance_descriptor for the live re-ID gallery this
    # same kind of hand-off can come from). Still only a handful of samples
    # accumulated so far, same as real life right after the hand-off.
    switcher.color_samples = [(114.0, 70.0, 100.0)] * 5
    tracker._team_assign_counter = 9
    tracker._assign_teams()
    assert switcher.team == home_label
    assert switcher.team_locked is False  # disagreed with the pending REFEREE read - still not locked

    # A third, later read agrees with the second - but still shouldn't lock
    # yet, since only a handful of color samples have accumulated so far (see
    # MIN_COLOR_SAMPLES_TO_LOCK's docstring for the separate failure mode this
    # guards against: a correctly-tracked player's first couple of reads can
    # themselves be a noisy outlier, so two agreeing reads alone isn't enough
    # unless there's also a real chunk of accumulated history behind them).
    tracker._team_assign_counter = 9
    tracker._assign_teams()
    assert switcher.team == home_label
    assert switcher.team_locked is False

    # Once enough real history has accumulated (still the same, consistent
    # navy/blue reading), the next agreeing re-cluster finally locks it in.
    switcher.color_samples = [(114.0, 70.0, 100.0)] * tracker.MIN_COLOR_SAMPLES_TO_LOCK
    tracker._team_assign_counter = 9
    tracker._assign_teams()
    assert switcher.team == home_label
    assert switcher.team_locked is True


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


def test_event_detector_goal_zone_uses_calibration_when_camera_shoots_down_pitch_length():
    # A camera that shoots down the length of the pitch (not along a sideline):
    # the far goal line sits near the TOP-MIDDLE of the frame, not at a left/right
    # edge. The near goal line (bottom of frame) is wide because it's close to the
    # camera; the far one (top of frame) is narrow and roughly centered.
    tl, tr, br, bl = (300.0, 650.0), (630.0, 120.0), (760.0, 120.0), (1050.0, 650.0)
    pitch_length_m, pitch_width_m = 40.0, 20.0
    calibration = PitchCalibration([tl, tr, br, bl], pitch_length_m, pitch_width_m)
    pitch_bounds = (300, 120, 1050, 650)

    # A point comfortably inside the real (HOME, world x=pitch_length_m) goal
    # mouth - near the top-middle of the frame, nowhere near its left/right edges.
    goal_px = calibration.world_to_pixel(39.0, 10.0)
    assert 500 < goal_px[0] < 900
    assert goal_px[1] < 300

    player = TrackedObject(1, "player", [0, 0, 0, 0], (50.0, 50.0), 1)
    player.team = "HOME"
    ball = TrackedObject(99, "ball", [0, 0, 0, 0], goal_px, 1)

    uncalibrated = EventDetector(frame_width=1280, frame_height=720, fps=30.0, pitch_bounds=pitch_bounds)
    calibrated = EventDetector(
        frame_width=1280, frame_height=720, fps=30.0, pitch_bounds=pitch_bounds, calibration=calibration
    )

    uncal_events, cal_events = [], []
    for frame_no in (1, 2):
        uncal_events += uncalibrated.process_frame([player, ball], frame_no=frame_no)
        cal_events += calibrated.process_frame([player, ball], frame_no=frame_no)

    # The old pitch-bounds/frame-edge heuristic's goal zone sits at the pixel
    # edges of the pitch bounding box - nowhere near this camera's real goal.
    assert [e for e in uncal_events if e.event_type == "GOAL"] == []
    # The calibrated goal zone, projected forward from the real goal-line
    # position, does catch it.
    goal_events = [e for e in cal_events if e.event_type == "GOAL"]
    assert len(goal_events) == 1
    assert goal_events[0].team == "HOME"


def test_calibration_contains_point_px_excludes_technical_area():
    # Regression test for real footage where color-based pitch segmentation
    # (app.cv.pitch_mask.PitchMask) bled into the technical area/dugout strip
    # next to the touchline - bench/coaching staff standing there got tracked
    # as "player" tracks and contaminated team pass/possession stats (see
    # PitchCalibration.contains_point_px's docstring). A calibrated camera's
    # real-world pitch rectangle gives a precise, color-independent second
    # check that isn't fooled by grass-colored ground just past the line.
    tl, tr, br, bl = (100.0, 600.0), (200.0, 100.0), (1080.0, 100.0), (1180.0, 600.0)
    pitch_length_m, pitch_width_m = 105.0, 68.0
    calibration = PitchCalibration([tl, tr, br, bl], pitch_length_m, pitch_width_m)

    # Comfortably inside the real pitch.
    inside_px = calibration.world_to_pixel(pitch_length_m / 2.0, pitch_width_m / 2.0)
    assert calibration.contains_point_px(*inside_px, margin_m=3.0) is True

    # 10m past the near touchline - a technical-area/bench distance, well
    # beyond even a generous margin, but still on grass-colored ground a
    # color mask alone can't tell apart from the pitch.
    bench_px = calibration.world_to_pixel(pitch_length_m / 2.0, pitch_width_m + 10.0)
    assert calibration.contains_point_px(*bench_px, margin_m=3.0) is False

    # A real throw-in taken right at the line should still be allowed through
    # by the margin, not clipped as if it were off-pitch.
    throw_in_px = calibration.world_to_pixel(pitch_length_m / 2.0, pitch_width_m + 1.0)
    assert calibration.contains_point_px(*throw_in_px, margin_m=3.0) is True


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


def test_shot_event_works_with_goal_axis_y_for_a_behind_the_goal_camera():
    # Regression test for a real gap found when validating against a
    # stationary camera mounted behind/above one goal, shooting down the
    # pitch length (Turflo-style) - goals sit at the top/bottom of frame,
    # not left/right. Without goal_axis="y", the default left/right zone
    # placement would put both "goals" at the sideline hoardings, and every
    # shot's direction/distance math (which routes through _to_world_m)
    # would scale along-pitch motion by the pitch WIDTH instead of its
    # LENGTH. This mirrors test_shot_event_carries_a_plausible_xg exactly,
    # just with the roles of x and y swapped.
    detector = EventDetector(frame_width=1280, frame_height=720, fps=30.0, goal_axis="y")

    home_player = TrackedObject(1, "player", [620, 480, 660, 520], (640.0, 500.0), 1)
    home_player.team = "HOME"
    ball_at_player = TrackedObject(99, "ball", [630, 490, 650, 510], (640.0, 500.0), 1)
    detector.process_frame([home_player, ball_at_player], frame_no=1)

    # 10 frames later, dead-central and deep in the (bottom/near) goal box,
    # moving fast and straight down toward it.
    ball_struck = TrackedObject(99, "ball", [630, 680, 650, 700], (640.0, 690.0), 11)
    events = detector.process_frame([home_player, ball_struck], frame_no=11)

    shot_evts = [e for e in events if e.event_type == "SHOT"]
    assert len(shot_evts) == 1
    assert shot_evts[0].team == "HOME"
    xg = shot_evts[0].data_json["xg"]
    assert 0.6 < xg <= 1.0


def test_shot_max_distance_is_a_fixed_real_world_cap_not_a_pitch_fraction():
    # Regression test for a real bug found via ground-truth video review on a
    # Barcelona vs Espanyol run (a real 105x68m pitch): the old cap was a
    # FRACTION of pitch_length_m (0.4), so it silently widened from ~16m on
    # the 40m turf pitch this was tuned against to ~42m on a full-size pitch -
    # every flagged "shot" in that run clustered at 22-42m, and a manual video
    # check at 33m showed calm, uncontested midfield build-up play, nowhere
    # near a shooting position. The cap must stay a fixed real-world distance
    # regardless of pitch size.
    px_to_m_x = 105.0 / 1920  # matches EventDetector's own linear fallback scale
    travel_px = (90.0 / 3.6 * (10 / 30.0)) / px_to_m_x  # ~90 km/h over the 10-frame window
    y = 540.0

    def fires_shot(distance_from_goal_m: float) -> bool:
        detector = EventDetector(
            frame_width=1920, frame_height=1080, fps=30.0,
            pitch_length_m=105.0, pitch_width_m=68.0,
        )
        goal_x = detector.home_target_goal[0]
        end_x = goal_x - distance_from_goal_m / px_to_m_x
        start_x = end_x - travel_px

        player_start = TrackedObject(1, "player", [start_x - 20, y - 20, start_x + 20, y + 20], (start_x, y), 1)
        player_start.team = "HOME"
        ball_start = TrackedObject(99, "ball", [start_x - 10, y - 10, start_x + 10, y + 10], (start_x, y), 1)
        detector.process_frame([player_start, ball_start], frame_no=1)

        player_end = TrackedObject(1, "player", [end_x - 20, y - 20, end_x + 20, y + 20], (end_x, y), 11)
        player_end.team = "HOME"
        ball_end = TrackedObject(99, "ball", [end_x - 10, y - 10, end_x + 10, y + 10], (end_x, y), 11)
        events = detector.process_frame([player_end, ball_end], frame_no=11)
        return any(e.event_type == "SHOT" for e in events)

    assert fires_shot(35.0) is False  # would have passed under the old 0.4*105=42m cap
    assert fires_shot(20.0) is True  # a genuine shooting-range strike, unaffected by the fix


def test_shot_with_implausible_speed_is_rejected_as_tracking_noise():
    # Regression test for a real bug found on production footage: most
    # detected "shots" were computed at 140-165 km/h, far beyond any real
    # football shot (the fastest officially recorded is ~131 km/h, by a
    # professional). That's the ball track briefly snapping onto a spurious
    # detection elsewhere in frame (noise, or a second ball/object) rather
    # than following the real ball - not a fast strike - and must be rejected.
    detector = EventDetector(frame_width=1280, frame_height=720, fps=30.0)

    home_player = TrackedObject(1, "player", [980, 340, 1020, 380], (1000.0, 360.0), 1)
    home_player.team = "HOME"
    ball_at_player = TrackedObject(99, "ball", [990, 350, 1010, 370], (1000.0, 360.0), 1)
    detector.process_frame([home_player, ball_at_player], frame_no=1)

    # Same end position/direction/timing as the real shot above, but starting
    # much farther away in the same 10-frame window - implying ~1400 km/h.
    detector._last_ball_pos = (-3000.0, 360.0)
    detector._last_ball_frame = 1
    ball_teleported = TrackedObject(99, "ball", [1190, 350, 1210, 370], (1200.0, 360.0), 11)
    events = detector.process_frame([home_player, ball_teleported], frame_no=11)

    assert [e for e in events if e.event_type == "SHOT"] == []


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


def test_zone_touches_flip_when_home_attacks_left():
    # Same fixture as above but for a match where HOME actually attacks the
    # LEFT (low-x) goal - every team's zone attribution must mirror, since
    # "attacking third" is relative to which goal a team is trying to score
    # in, not a fixed side of the frame.
    detector = EventDetector(
        frame_width=1200, frame_height=600, fps=30.0, pitch_bounds=(0, 0, 1200, 600),
        home_attacks_right=False,
    )

    def touch(track_id, team, x, frame_no):
        player = TrackedObject(track_id, "player", [x - 20, 530, x + 20, 570], (float(x), 550.0), frame_no)
        player.team = team
        ball = TrackedObject(99, "ball", [x - 10, 545, x + 10, 555], (float(x), 550.0), frame_no)
        detector.process_frame([player, ball], frame_no=frame_no)

    touch(1, "HOME", 1000, 1)   # HOME's own defensive third (now the RIGHT side)
    touch(1, "HOME", 600, 101)  # midfield
    touch(1, "HOME", 100, 201)  # HOME's attacking third (now the LEFT side)
    touch(2, "AWAY", 100, 301)  # AWAY's own defensive third (mirrored)
    touch(2, "AWAY", 1000, 401)  # AWAY's attacking third (mirrored)

    assert detector.zone_touches["HOME"] == {"DEFENSIVE": 1, "MIDDLE": 1, "ATTACKING": 1}
    assert detector.zone_touches["AWAY"] == {"DEFENSIVE": 1, "MIDDLE": 0, "ATTACKING": 1}


def test_goal_scoring_team_respects_home_attacks_right_false():
    # Regression test for a real bug found via ground-truth video review on
    # Rock8 M1: the hardcoded "HOME always attacks right" assumption silently
    # swaps GOAL scoring-team attribution for any match where that's not true -
    # confirmed here on the same simple pitch_bounds geometry as the corner
    # tests above, since this shares the identical convention.
    pitch_bounds = (0, 0, 1200, 600)
    bystander = TrackedObject(1, "player", [0, 550, 40, 590], (20.0, 570.0), 1)
    bystander.team = "HOME"

    default_detector = EventDetector(frame_width=1200, frame_height=600, fps=30.0, pitch_bounds=pitch_bounds)
    flipped_detector = EventDetector(
        frame_width=1200, frame_height=600, fps=30.0, pitch_bounds=pitch_bounds, home_attacks_right=False,
    )

    default_events, flipped_events = [], []
    for frame_no in (1, 2):
        ball = TrackedObject(99, "ball", [1150, 290, 1170, 310], (1160.0, 300.0), frame_no)  # right goal
        default_events += default_detector.process_frame([bystander, ball], frame_no=frame_no)
        flipped_events += flipped_detector.process_frame([bystander, ball], frame_no=frame_no)

    default_goal = next(e for e in default_events if e.event_type == "GOAL")
    flipped_goal = next(e for e in flipped_events if e.event_type == "GOAL")
    assert default_goal.team == "HOME"  # default convention: HOME attacks right
    assert flipped_goal.team == "AWAY"  # flipped: AWAY attacks right instead


def test_event_detector_update_pitch_bounds_moves_goal_zone_with_a_panning_camera():
    # Regression test for a real bug found on Rock8 M1 real footage: the goal
    # zone was computed ONCE from wherever the pitch happened to sit in a
    # single frame, then reused for the whole match even though the camera
    # pans throughout. On real footage this produced two false "goals" (a
    # stale, wrong-side box happened to catch a fast ball pass-through) and
    # missed a real one (the true goal had moved elsewhere on screen by the
    # time it was scored). update_pitch_bounds lets a caller refresh the zone
    # mid-match instead of freezing it at construction time.
    detector = EventDetector(frame_width=1200, frame_height=600, fps=30.0, pitch_bounds=(0, 0, 1200, 600))
    player = TrackedObject(1, "player", [0, 550, 40, 590], (20.0, 570.0), 1)
    player.team = "HOME"

    # (630, 300) is nowhere near either goal zone under the ORIGINAL bounds
    # (left_goal x:[0,72], right_goal x:[1128,1200]) - no goal should fire.
    for frame_no in (1, 2):
        ball = TrackedObject(99, "ball", [620, 290, 640, 310], (630.0, 300.0), frame_no)
        events = detector.process_frame([player, ball], frame_no=frame_no)
    assert [e for e in events if e.event_type == "GOAL"] == []

    # The camera pans right by 600px - the SAME physical ball position (630,
    # 300) now sits inside the (shifted) left goal zone [600,672].
    detector.update_pitch_bounds((600, 0, 1800, 600))
    for frame_no in (3, 4):
        ball = TrackedObject(99, "ball", [620, 290, 640, 310], (630.0, 300.0), frame_no)
        events = detector.process_frame([player, ball], frame_no=frame_no)

    goal_events = [e for e in events if e.event_type == "GOAL"]
    assert len(goal_events) == 1
    assert goal_events[0].data_json["goal_side"] == "LEFT"


def test_detect_corners_uses_pitch_bounds_history_for_a_panning_camera():
    # Companion regression test for _detect_corners: a reappearance point that
    # is nowhere near a corner under the STALE initial pitch_bounds, but
    # clearly is once the camera has panned (per pitch_bounds_history), must
    # be attributed using the bounds sample that was actually current at the
    # reappearance frame - not the single static box.
    stale_bounds = (0, 0, 1200, 600)
    panned_bounds = (600, 0, 1800, 600)

    without_history = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (300.0, 300.0), 200: (650.0, 40.0)},
        pitch_bounds=stale_bounds,
        frame_width=1200, frame_height=600, fps=30.0, max_gap_ball=15,
    )
    assert without_history == []  # (650, 40) isn't near any corner of the stale box

    with_history = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (300.0, 300.0), 200: (650.0, 40.0)},
        pitch_bounds=stale_bounds,
        frame_width=1200, frame_height=600, fps=30.0, max_gap_ball=15,
        pitch_bounds_history=[(0, stale_bounds), (150, panned_bounds)],
    )
    assert len(with_history) == 1
    assert with_history[0].team == "AWAY"  # near the (panned) left goal's corner


def test_detect_throw_ins_credits_the_team_that_did_not_put_it_out():
    pitch_bounds = (0, 0, 1200, 600)

    events = _detect_throw_ins(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (600.0, 300.0), 200: (600.0, 40.0)},  # top touchline, mid-length
        pitch_bounds=pitch_bounds,
        frame_width=1200,
        frame_height=600,
        fps=30.0,
        max_gap_ball=15,
        possessor_team_by_frame={8: "HOME"},  # last touched by HOME just before it went out
    )

    assert len(events) == 1
    assert events[0].event_type == "THROW_IN"
    assert events[0].team == "AWAY"  # HOME put it out, so AWAY throws it back in


def test_detect_throw_ins_ignores_ordinary_sampling_gaps():
    events = _detect_throw_ins(
        ball_sorted_frames=[10, 20],  # gap of 10, under max_gap_ball
        ball_positions={10: (600.0, 300.0), 20: (600.0, 40.0)},
        pitch_bounds=(0, 0, 1200, 600),
        frame_width=1200, frame_height=600, fps=30.0, max_gap_ball=15,
        possessor_team_by_frame={8: "HOME"},
    )
    assert events == []


def test_detect_throw_ins_does_not_fire_for_a_corner_shaped_reappearance():
    # Near BOTH a touchline and a goal-line end at once is a corner, not a
    # throw-in (see _detect_corners) - the two detectors must not double-count
    # the same real stoppage as both events.
    events = _detect_throw_ins(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (600.0, 300.0), 200: (40.0, 40.0)},  # top-left corner
        pitch_bounds=(0, 0, 1200, 600),
        frame_width=1200, frame_height=600, fps=30.0, max_gap_ball=15,
        possessor_team_by_frame={8: "HOME"},
    )
    assert events == []


def test_detect_throw_ins_skips_when_last_possessor_is_unknown():
    # No real player was ever recorded as possessing the ball beforehand (e.g.
    # the ball went out before anyone was tracked nearby) - can't credit a
    # team we don't actually know, so this must not fabricate an attribution.
    events = _detect_throw_ins(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (600.0, 300.0), 200: (600.0, 40.0)},
        pitch_bounds=(0, 0, 1200, 600),
        frame_width=1200, frame_height=600, fps=30.0, max_gap_ball=15,
        possessor_team_by_frame={},
    )
    assert events == []


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


def test_detect_corners_respects_home_attacks_right_false():
    # Regression test for a real bug found via ground-truth video review: which
    # team attacks which goal is a per-match camera/kickoff fact, not something
    # derivable from pitch geometry - a match where HOME actually attacks the
    # LEFT goal must flip corner attribution accordingly, not silently assume
    # HOME always attacks right (the old, hardcoded behavior would have
    # credited every corner in this match to the wrong team).
    pitch_bounds = (0, 0, 1200, 600)

    events = _detect_corners(
        ball_sorted_frames=[10, 200],
        ball_positions={10: (600.0, 300.0), 200: (1150.0, 50.0)},  # top-right corner
        pitch_bounds=pitch_bounds,
        frame_width=1200,
        frame_height=600,
        fps=30.0,
        max_gap_ball=15,
        home_attacks_right=False,
    )

    assert len(events) == 1
    assert events[0].team == "AWAY"  # AWAY attacks right in this match, so it's their corner


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
