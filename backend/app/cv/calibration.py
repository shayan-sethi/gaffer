import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PitchCalibration:
    """
    Perspective-correct pixel -> real-world-meters mapping via a homography
    computed from 4 manually-marked pitch corners.

    This replaces the flat `pitch_length_m / pitch_px_width` linear ratio used
    elsewhere in the CV pipeline, which is only accurate at the depth those
    reference pixels happen to sit at. On a fixed, angled camera (Gaffer's
    setup), a meter near the camera spans far more pixels than a meter at the
    far side of the pitch, so a single linear ratio systematically over- or
    under-states distance/speed depending on where on the pitch play happens.
    A homography corrects for that by mapping every pixel through the same
    perspective transform the camera itself applied.
    """

    def __init__(
        self,
        image_points: Sequence[Tuple[float, float]],
        pitch_length_m: float,
        pitch_width_m: float,
    ):
        """
        `image_points` must be exactly 4 pixel (x, y) points marking the pitch
        corners in TL, TR, BR, BL order - i.e. matching the (0,0)/(pitch_length_m,
        0)/(pitch_length_m, pitch_width_m)/(0, pitch_width_m) real-world corners
        (the same left=0/top=0 convention StatsCalculator and EventDetector
        already use for pitch_bounds).
        """
        if len(image_points) != 4:
            raise ValueError(f"Calibration requires exactly 4 pitch corner points, got {len(image_points)}.")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(
            [
                [0.0, 0.0],
                [pitch_length_m, 0.0],
                [pitch_length_m, pitch_width_m],
                [0.0, pitch_width_m],
            ],
            dtype=np.float32,
        )
        homography, _ = cv2.findHomography(src, dst)
        if homography is None:
            raise ValueError("Could not compute a homography from the given calibration points (are they collinear?).")
        self._homography = homography
        try:
            self._inverse_homography = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            raise ValueError("Could not invert the calibration homography (are the points collinear?).")
        self._pitch_length_m = pitch_length_m
        self._pitch_width_m = pitch_width_m

    def pixel_to_world_m(self, x: float, y: float) -> Tuple[float, float]:
        """Maps a single pixel coordinate to real-world pitch meters (x, y)."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._homography)
        return float(world[0, 0, 0]), float(world[0, 0, 1])

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Inverse of pixel_to_world_m: maps a real-world pitch-meters coordinate
        (x, y) to the pixel it projects to under this camera's calibration. Used
        to place features whose real-world position is known by definition (e.g.
        a goal sits at the real goal line) rather than derived from a detection,
        so there's no pixel coordinate for it until it's projected forward."""
        pt = np.array([[[x_m, y_m]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, self._inverse_homography)
        return float(px[0, 0, 0]), float(px[0, 0, 1])

    def contains_point_px(self, x_px: float, y_px: float, margin_m: float = 0.0) -> bool:
        """
        True if this pixel maps to a point within the real pitch rectangle
        (0..pitch_length_m x 0..pitch_width_m), padded by `margin_m` in every
        direction.

        Used to reject "player" detections that a color-based grass mask alone
        can't reliably exclude - confirmed on real footage (a professional
        match's technical area/dugout strip) that bench/coaching staff
        standing on grass-colored ground just past the touchline get through
        app.cv.pitch_mask.PitchMask's color segmentation, since there's no
        hard color break at the touchline itself for it to key off. This is a
        precise geometric check instead - independent of lighting, turf color,
        or how close the technical area's ground color is to the pitch's -
        but only available once a camera has been calibrated (see
        PitchCalibration.from_points_json); an uncalibrated camera has no
        pitch rectangle to test against, so callers should fall back to the
        color-based mask alone in that case.
        """
        x_m, y_m = self.pixel_to_world_m(x_px, y_px)
        return (
            -margin_m <= x_m <= self._pitch_length_m + margin_m
            and -margin_m <= y_m <= self._pitch_width_m + margin_m
        )

    def distance_m(self, p1_px: Tuple[float, float], p2_px: Tuple[float, float]) -> float:
        """Real-world straight-line distance in meters between two pixel points."""
        x1, y1 = self.pixel_to_world_m(*p1_px)
        x2, y2 = self.pixel_to_world_m(*p2_px)
        return float(np.hypot(x2 - x1, y2 - y1))

    @classmethod
    def from_points_json(
        cls,
        points: Optional[List[Sequence[float]]],
        pitch_length_m: float,
        pitch_width_m: float,
    ) -> Optional["PitchCalibration"]:
        """
        Builds a PitchCalibration from the raw JSON-serializable point list
        stored on Camera.calibration_points (or carried through a Celery task
        payload), or returns None if the camera hasn't been calibrated / the
        points are malformed - callers should fall back to the linear
        approximation in that case, not fail the whole pipeline over it.
        """
        if not points:
            return None
        try:
            image_points = [(float(p[0]), float(p[1])) for p in points]
            return cls(image_points, pitch_length_m, pitch_width_m)
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Ignoring invalid camera calibration points: {e}")
            return None
