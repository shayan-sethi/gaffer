import logging
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PitchMask:
    """
    Derives a rough field-of-play mask via grass-color segmentation, computed once
    from a sample frame since Gaffer cameras are fixed for the whole match. This is
    what keeps sideline spectators, benches, and bystanders - which YOLO's generic
    "person" class detects just as happily as players - from being tracked as
    players and fragmenting/inflating track IDs.
    """

    def __init__(self, frame: np.ndarray):
        self.height, self.width = frame.shape[:2]
        self.mask = self._compute_mask(frame)
        if self.mask is None:
            logger.warning("Could not confidently segment a pitch region; player filtering by ROI is disabled.")

    def _compute_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad green range to tolerate turf color, day/night lighting, and shadows
        lower = np.array([25, 20, 20])
        upper = np.array([95, 255, 255])
        grass = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((15, 15), np.uint8)
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, kernel)
        grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        # Require the pitch to cover a meaningful chunk of the frame, otherwise
        # distrust the segmentation rather than risk filtering out real players.
        if cv2.contourArea(largest) < 0.15 * self.width * self.height:
            return None

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        # Small dilation so players standing right at the touchline aren't clipped -
        # kept tight, since a generous dilation lets sideline bench/bystanders leak in.
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        return mask

    def contains_point(self, x: float, y: float) -> bool:
        if self.mask is None:
            return True  # Segmentation wasn't confident - don't filter anything
        xi = int(np.clip(x, 0, self.width - 1))
        yi = int(np.clip(y, 0, self.height - 1))
        return bool(self.mask[yi, xi])
