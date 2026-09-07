import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Same acceptance threshold CentroidTracker's live re-ID gallery uses for this
# exact appearance signal (see CentroidTracker._appearance_descriptor /
# _hist_distance / reid_max_distance) - a fragment pair below this is
# confidently the same person and gets auto-merged. A pair between this and
# AMBIGUOUS_MAX_DISTANCE is plausible but not certain, and gets surfaced for a
# human call instead of silently merged - a wrong merge would corrupt two real
# players' stats together, which is worse than leaving a fragment unmerged.
CONFIDENT_MAX_DISTANCE = 0.3
AMBIGUOUS_MAX_DISTANCE = 0.45

# No real player covers ground faster than this between two sightings - the
# same sanity cap already used elsewhere in this codebase for player top
# speed (see StatsCalculator / EventDetector.SHOT_MAX_SPEED_KMH). A gap that
# would require exceeding it can't be the same person, however similar their
# appearance looks - two different same-kit teammates can score a deceptively
# low appearance distance (see hist_distance's docstring).
MAX_PLAUSIBLE_SPEED_KMH = 35.0


def hist_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """
    Hellinger distance between two normalized appearance histograms (0 =
    identical, 1 = disjoint) - the exact metric CentroidTracker's live re-ID
    gallery uses (see CentroidTracker._hist_distance), reused here so a
    fragment-merge decision is judged by the same signal a live re-ID match
    would have been. Note this alone can't fully disambiguate two teammates
    in identical kits (same known limitation as live re-ID) - that's why
    merges here are also gated on physical plausibility (see MergeCandidate).

    This formula (the Bhattacharyya coefficient, sum of sqrt(a_i*b_i)) is
    only mathematically guaranteed to land in [0, 1] when `a` and `b` are
    true probability distributions - L1-normalized (sums to 1), not
    L2-normalized (unit Euclidean length). Callers MUST pass L1-normalized
    descriptors (see appearance_descriptor_from_crop). A previous version of
    that function L2-normalized instead, which let a real (non-degenerate)
    photo's histogram score a Bhattacharyya coefficient well above 1 -
    clipped by max(0.0, ...) to a distance of 0 regardless of what was
    actually being compared, silently making every real crop read as
    "identical" to every other. That bug lived in CentroidTracker's live
    re-ID descriptor too (same formula, same L2 normalization) for this
    entire codebase's history until now, which means the appearance check in
    the live re-ID gallery was effectively a no-op on real footage - only the
    position gate was ever actually discriminating between candidates.
    """
    if a is None or b is None:
        return float("inf")
    bc = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
    return math.sqrt(max(0.0, 1.0 - bc))


class TrackFragment:
    """
    One tracker-assigned identity awaiting consolidation against the known
    real roster size - may already be a whole real player, or just one piece
    of a real player who was lost mid-match and re-spawned as a new track ID.
    """

    def __init__(
        self,
        track_id: int,
        team: str,
        history: List[Tuple[int, Tuple[float, float]]],
        appearance_hist: Optional[np.ndarray] = None,
    ):
        if not history:
            raise ValueError(f"TrackFragment {track_id} has no history")
        self.track_id = track_id
        self.team = team
        self.history = history
        self.appearance_hist = appearance_hist

    @property
    def first_frame(self) -> int:
        return self.history[0][0]

    @property
    def last_frame(self) -> int:
        return self.history[-1][0]

    @property
    def first_pos(self) -> Tuple[float, float]:
        return self.history[0][1]

    @property
    def last_pos(self) -> Tuple[float, float]:
        return self.history[-1][1]


def _overlaps(a: TrackFragment, b: TrackFragment) -> bool:
    """True if both fragments were active at the same time - impossible for
    the same real person, so this rules a pair out before appearance is even
    checked, and later guards against an invalid transitive merge through a
    third fragment (see consolidate)."""
    return a.first_frame <= b.last_frame and b.first_frame <= a.last_frame


def _bridge_speed_kmh(a: TrackFragment, b: TrackFragment, fps: float, px_to_m: float) -> float:
    """Implied average speed (km/h) to cover the gap between whichever
    fragment ends first and whichever starts next."""
    earlier, later = (a, b) if a.last_frame <= b.first_frame else (b, a)
    frame_gap = later.first_frame - earlier.last_frame
    dist_m = math.hypot(
        later.first_pos[0] - earlier.last_pos[0], later.first_pos[1] - earlier.last_pos[1]
    ) * px_to_m
    if frame_gap <= 0:
        return float("inf") if dist_m > 0 else 0.0
    seconds = frame_gap / fps
    return (dist_m / seconds) * 3.6


class MergeCandidate:
    def __init__(self, a: TrackFragment, b: TrackFragment, distance: float, speed_kmh: float):
        self.a = a
        self.b = b
        self.distance = distance
        self.speed_kmh = speed_kmh

    @property
    def confident(self) -> bool:
        return self.distance < CONFIDENT_MAX_DISTANCE

    @property
    def ambiguous(self) -> bool:
        return CONFIDENT_MAX_DISTANCE <= self.distance < AMBIGUOUS_MAX_DISTANCE


def find_candidates(fragments: List[TrackFragment], fps: float, px_to_m: float) -> List[MergeCandidate]:
    """All same-team, non-overlapping, physically-plausible-gap fragment
    pairs that could be the same real player, best (closest appearance) first."""
    candidates = []
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            a, b = fragments[i], fragments[j]
            if a.team != b.team or _overlaps(a, b):
                continue
            speed = _bridge_speed_kmh(a, b, fps, px_to_m)
            if speed > MAX_PLAUSIBLE_SPEED_KMH:
                continue
            distance = hist_distance(a.appearance_hist, b.appearance_hist)
            candidates.append(MergeCandidate(a, b, distance, speed))
    candidates.sort(key=lambda c: c.distance)
    return candidates


class ConsolidationResult:
    def __init__(self):
        # Each inner list is the set of track_ids merged into one real identity.
        self.groups: List[List[int]] = []
        # Confident-adjacent-but-not-quite candidates left for a human to
        # confirm or reject - never auto-applied.
        self.ambiguous: List[MergeCandidate] = []


def consolidate(fragments: List[TrackFragment], fps: float, px_to_m: float) -> ConsolidationResult:
    """
    Greedily auto-merges only CONFIDENT same-team fragment pairs (appearance
    distance below the live re-ID system's own acceptance threshold, and a
    physically plausible gap) into single identities, best matches first.
    Anything short of that bar is left alone and surfaced as an ambiguous
    candidate instead of silently guessed at.
    """
    by_id = {f.track_id: f for f in fragments}
    groups: Dict[int, List[int]] = {f.track_id: [f.track_id] for f in fragments}
    owner: Dict[int, int] = {f.track_id: f.track_id for f in fragments}

    result = ConsolidationResult()
    for c in find_candidates(fragments, fps, px_to_m):
        ga, gb = owner[c.a.track_id], owner[c.b.track_id]
        if ga == gb:
            continue  # already the same identity via an earlier, better link

        group_a_ids, group_b_ids = groups[ga], groups[gb]
        # A transitive merge through a third fragment must still hold up to
        # the same hard physical constraint a direct pairwise candidate
        # would - reject outright if it would put any two mutually-
        # overlapping fragments into the same identity (appearance alone
        # can be deceptively similar between two teammates in identical
        # kits - see hist_distance's docstring).
        would_overlap = any(_overlaps(by_id[x], by_id[y]) for x in group_a_ids for y in group_b_ids)
        if would_overlap:
            continue

        if c.confident:
            merged = group_a_ids + group_b_ids
            for tid in merged:
                owner[tid] = ga
            groups[ga] = merged
            del groups[gb]
        elif c.ambiguous:
            result.ambiguous.append(c)

    result.groups = list(groups.values())
    return result


def _region_hist(crop: np.ndarray, y1_frac: float, y2_frac: float) -> np.ndarray:
    h = crop.shape[0]
    ya, yb = int(h * y1_frac), int(h * y2_frac)
    region = crop[ya:yb]
    if region.size == 0:
        return np.zeros(64, dtype=np.float32)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    # L1 (sum-to-1), not L2 - see hist_distance's docstring for why this
    # matters: its Bhattacharyya-coefficient formula only stays in [0, 1] for
    # true probability distributions (L1-normalized), and silently collapses
    # to "everything looks identical" (distance 0) for L2-normalized ones on
    # any sufficiently busy real crop.
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()


def appearance_descriptor_from_crop(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Same two-region (torso + legs) HSV histogram descriptor CentroidTracker
    uses for live re-ID (see CentroidTracker._appearance_descriptor), applied
    to an already-cropped player image rather than a full frame+bbox - used
    to backfill an appearance signature for tracks from a run that didn't
    persist one (see sample_appearance_from_video).
    """
    if crop is None or crop.size == 0:
        return None
    torso = _region_hist(crop, 0.15, 0.55)
    legs = _region_hist(crop, 0.55, 0.95)
    descriptor = np.concatenate([torso, legs]).astype(np.float32)
    # L1-normalize the combined vector too, so it's one valid probability
    # distribution over both regions together (each already sums to 1, so
    # the raw concatenation sums to 2).
    total = descriptor.sum()
    return descriptor / total if total > 0 else descriptor


def sample_appearance_from_video(
    cap: cv2.VideoCapture,
    history: List[Tuple[int, Tuple[float, float]]],
    box_half_width: float = 25.0,
    box_height: float = 140.0,
    num_samples: int = 3,
) -> Optional[np.ndarray]:
    """
    Backfills an appearance descriptor for a fragment from a completed
    tracking run that didn't persist one - track.py only saves each track's
    (frame_no, centroid) history, not a real per-frame bbox or appearance
    histogram. Reads a few representative frames straight from the source
    video and crops an approximate player box around the tracked centroid
    (bottom-center convention: the box sits ABOVE the centroid, not around
    it). This is inherently approximate - the real detection width/height
    wasn't saved - a live tracking pass has the real box and should prefer
    computing appearance_hist directly, the way CentroidTracker already does.
    """
    if not history:
        return None
    n = len(history)
    sample_idxs = sorted({0, n // 2, n - 1})[:num_samples]
    descriptors = []
    for idx in sample_idxs:
        frame_no, (cx, cy) = history[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            continue
        x1, y1 = max(0, int(cx - box_half_width)), max(0, int(cy - box_height))
        x2, y2 = min(frame.shape[1], int(cx + box_half_width)), min(frame.shape[0], int(cy))
        descriptor = appearance_descriptor_from_crop(frame[y1:y2, x1:x2])
        if descriptor is not None:
            descriptors.append(descriptor)
    if not descriptors:
        return None
    avg = np.mean(descriptors, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def assign_roster_numbers(
    groups: List[List[int]], roster_size: int
) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Assigns each consolidated identity a stable number 1..N within its team,
    ordered by when it first appeared - NOT a real printed jersey number (this
    pipeline has no way to read one), just a consistent label for reporting.
    Returns (track_id -> assigned number, groups actually numbered). If there
    are more groups than roster_size, the extra (least-confident / shortest)
    groups are left out of the numbering and returned separately so the
    caller can surface "N unresolved identities" rather than silently forcing
    a fit that isn't there.
    """
    # Longer-lived groups are more likely a real, distinct player rather than
    # a stray fragment (a detector false positive, or someone briefly in
    # frame) - sort the most-substantial groups first so those are the ones
    # that get numbered when there are more groups than roster slots.
    ordered = sorted(groups, key=lambda g: -sum(1 for _ in g))
    numbered, unresolved = ordered[:roster_size], ordered[roster_size:]
    numbered.sort(key=lambda g: min(g))  # stable, presentation order

    assignment: Dict[int, int] = {}
    for number, group in enumerate(numbered, start=1):
        for track_id in group:
            assignment[track_id] = number
    return assignment, unresolved
