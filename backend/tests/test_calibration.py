import cv2
import numpy as np
import pytest
from app.cv.calibration import PitchCalibration

PITCH_LENGTH_M = 40.0
PITCH_WIDTH_M = 20.0

# A trapezoid modeling a real fixed, angled camera: the near touchline (large y,
# close to camera) spans far more pixels than the far touchline (small y) even
# though both represent the same 40m real-world length.
TL = (400.0, 100.0)   # far-left
TR = (880.0, 100.0)   # far-right   (far edge: 480px for 40m -> 12 px/m)
BR = (1230.0, 650.0)  # near-right
BL = (50.0, 650.0)    # near-left   (near edge: 1180px for 40m -> 29.5 px/m)


def _forward_homography():
    """The ground-truth world->pixel projection these 4 corners were sampled
    from - used to forward-project arbitrary world points into pixel space so
    tests can check recovery of points *other* than the 4 calibration corners
    themselves."""
    world = np.array(
        [[0.0, 0.0], [PITCH_LENGTH_M, 0.0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0.0, PITCH_WIDTH_M]],
        dtype=np.float32,
    )
    image = np.array([TL, TR, BR, BL], dtype=np.float32)
    return cv2.getPerspectiveTransform(world, image)


def _world_to_pixel(forward_h, x_m, y_m):
    pt = np.array([[[x_m, y_m]]], dtype=np.float32)
    px = cv2.perspectiveTransform(pt, forward_h)
    return float(px[0, 0, 0]), float(px[0, 0, 1])


def test_calibration_corners_map_to_exact_world_positions():
    calib = PitchCalibration([TL, TR, BR, BL], PITCH_LENGTH_M, PITCH_WIDTH_M)

    assert calib.pixel_to_world_m(*TL) == pytest.approx((0.0, 0.0), abs=1e-3)
    assert calib.pixel_to_world_m(*TR) == pytest.approx((PITCH_LENGTH_M, 0.0), abs=1e-3)
    assert calib.pixel_to_world_m(*BR) == pytest.approx((PITCH_LENGTH_M, PITCH_WIDTH_M), abs=1e-3)
    assert calib.pixel_to_world_m(*BL) == pytest.approx((0.0, PITCH_WIDTH_M), abs=1e-3)


def test_calibration_corrects_perspective_scale_for_equal_real_distances():
    # Two genuinely-5m-apart pairs of world points, one near the far (narrow,
    # low px/m) touchline and one near the near (wide, high px/m) touchline,
    # forward-projected into pixel space through the ground-truth camera model.
    forward_h = _forward_homography()
    far_p1_px = _world_to_pixel(forward_h, 10.0, 1.0)
    far_p2_px = _world_to_pixel(forward_h, 15.0, 1.0)
    near_p1_px = _world_to_pixel(forward_h, 10.0, 19.0)
    near_p2_px = _world_to_pixel(forward_h, 15.0, 19.0)

    # A single flat pixel-to-meter ratio cannot be right for both depths.
    far_px_dist = np.hypot(far_p2_px[0] - far_p1_px[0], far_p2_px[1] - far_p1_px[1])
    near_px_dist = np.hypot(near_p2_px[0] - near_p1_px[0], near_p2_px[1] - near_p1_px[1])
    assert near_px_dist > far_px_dist * 1.5  # near touchline is visibly "wider" in pixels

    calib = PitchCalibration([TL, TR, BR, BL], PITCH_LENGTH_M, PITCH_WIDTH_M)

    # But the calibrated, perspective-corrected distance recovers ~5m at both depths.
    assert calib.distance_m(far_p1_px, far_p2_px) == pytest.approx(5.0, abs=0.05)
    assert calib.distance_m(near_p1_px, near_p2_px) == pytest.approx(5.0, abs=0.05)


def test_calibration_requires_exactly_four_points():
    with pytest.raises(ValueError):
        PitchCalibration([TL, TR, BR], PITCH_LENGTH_M, PITCH_WIDTH_M)


def test_calibration_rejects_degenerate_collinear_points():
    collinear = [(0.0, 100.0), (100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]
    with pytest.raises(ValueError):
        PitchCalibration(collinear, PITCH_LENGTH_M, PITCH_WIDTH_M)


def test_from_points_json_returns_none_when_uncalibrated():
    assert PitchCalibration.from_points_json(None, PITCH_LENGTH_M, PITCH_WIDTH_M) is None
    assert PitchCalibration.from_points_json([], PITCH_LENGTH_M, PITCH_WIDTH_M) is None


def test_from_points_json_returns_none_for_malformed_points_instead_of_raising():
    # Wrong point count and non-numeric points are real possibilities coming off
    # a JSON round-trip through a Celery task payload - this must degrade to the
    # linear fallback, not crash the whole tracking/stats pipeline.
    assert PitchCalibration.from_points_json([[1, 2], [3, 4], [5, 6]], PITCH_LENGTH_M, PITCH_WIDTH_M) is None
    assert PitchCalibration.from_points_json(
        [["a", "b"], [1, 2], [3, 4], [5, 6]], PITCH_LENGTH_M, PITCH_WIDTH_M
    ) is None


def test_from_points_json_builds_working_calibration_from_valid_points():
    calib = PitchCalibration.from_points_json([list(TL), list(TR), list(BR), list(BL)], PITCH_LENGTH_M, PITCH_WIDTH_M)
    assert calib is not None
    assert calib.pixel_to_world_m(*TL) == pytest.approx((0.0, 0.0), abs=1e-3)
