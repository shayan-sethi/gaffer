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

    def pixel_to_world_m(self, x: float, y: float) -> Tuple[float, float]:
        """Maps a single pixel coordinate to real-world pitch meters (x, y)."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._homography)
        return float(world[0, 0, 0]), float(world[0, 0, 1])

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
