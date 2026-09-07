from typing import Dict, List, Optional

import numpy as np

from app.cv.roster_consolidation import appearance_descriptor_from_crop, hist_distance

# Same acceptance threshold used everywhere else in this codebase for this
# exact appearance signal (see CentroidTracker.reid_max_distance and
# roster_consolidation.CONFIDENT_MAX_DISTANCE) - a match below this is
# confident; anything else is reported as unidentified rather than guessed.
DEFAULT_MAX_DISTANCE = 0.35


class PlayerGallery:
    """
    A fixed, closed-set roster of known real players, each represented by a
    reference appearance descriptor (built from one or more confirmed crops -
    see build_reference). Matching a new detection against this gallery,
    independently every frame, is how per-player identity survives a track
    that silently swaps onto a different real person mid-match (see
    roster_consolidation's module docstring for why continuous tracking alone
    can't be trusted here) - there's no "track" to swap once every sighting is
    re-identified fresh against a known reference instead of inherited from
    the previous frame's decision.
    """

    def __init__(self):
        self.references: Dict[str, np.ndarray] = {}

    def add_reference(self, label: str, descriptor: np.ndarray) -> None:
        self.references[label] = descriptor

    def build_reference(self, label: str, crops: List[np.ndarray]) -> None:
        """Builds a reference descriptor for `label` by averaging the
        descriptor of each given crop - multiple crops (different lighting/
        pose) make the reference more robust than any single sighting."""
        descriptors = [d for d in (appearance_descriptor_from_crop(c) for c in crops) if d is not None]
        if not descriptors:
            return
        avg = np.mean(descriptors, axis=0)
        # L1, not L2 - the mean of several L1-normalized (sum-to-1)
        # descriptors already sums to ~1 and only needs a final tidy-up, but
        # re-normalizing by L2 norm here would silently reintroduce the exact
        # bug hist_distance's docstring describes (a real descriptor's
        # Bhattacharyya coefficient against an L2-renormalized reference can
        # exceed 1, collapsing the distance to 0 regardless of the real
        # match). Every stored descriptor must stay L1-normalized.
        total = avg.sum()
        self.add_reference(label, avg / total if total > 0 else avg)

    def identify(
        self, descriptor: Optional[np.ndarray], max_distance: float = DEFAULT_MAX_DISTANCE
    ) -> Optional[str]:
        """Best-matching known identity for `descriptor`, or None if nothing
        in the gallery is close enough to trust - an unmatched sighting
        should be reported as unidentified, not forced onto the nearest
        (possibly wrong) reference."""
        if descriptor is None or not self.references:
            return None
        best_label, best_dist = None, max_distance
        for label, ref in self.references.items():
            dist = hist_distance(descriptor, ref)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    def identify_crop(
        self, crop: Optional[np.ndarray], max_distance: float = DEFAULT_MAX_DISTANCE
    ) -> Optional[str]:
        return self.identify(appearance_descriptor_from_crop(crop), max_distance)
