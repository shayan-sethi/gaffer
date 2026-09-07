import math
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.cv.detector import DetectionResult


class TrackedObject:
    def __init__(self, track_id: int, class_name: str, bbox: List[float], centroid: Tuple[float, float], frame_no: int):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.centroid = centroid  # (cx, cy) - usually bottom-center for players
        self.velocity: Tuple[float, float] = (0.0, 0.0)  # smoothed per-frame (dx, dy), used to predict through misses
        self.history: List[Tuple[int, Tuple[float, float]]] = [(frame_no, centroid)]  # (frame_no, centroid)
        self.consecutive_lost = 0
        # Real successful matches (the initial spawn counts as 1, .update() counts
        # the rest) - see CentroidTracker.CONFIRM_HITS for why a track needs more
        # than 1 before it's allowed into the re-ID gallery.
        self.hits = 1
        self.team: str = "UNASSIGNED"  # HOME, AWAY, REFEREE
        # Once True, _assign_teams leaves .team alone - a real player doesn't
        # switch sides mid-match, so once a track has been confidently labeled
        # (by color clustering or the position fallback), later periodic
        # re-clusters must not be able to flip it. Before this, a track's label
        # could drift across a cluster boundary (lighting/shadow/exposure
        # changes) and silently misattribute goals/shots/possession to the
        # wrong team mid-match.
        self.team_locked: bool = False
        self.total_distance_px: float = 0.0
        self.color_samples: List[Tuple[float, float, float]] = []  # accumulated mean BGR jersey samples
        self.appearance_hist: Optional[np.ndarray] = None  # EMA'd re-ID descriptor (see CentroidTracker._appearance_descriptor)
        self.lost_at_frame: Optional[int] = None  # frame_no this track was moved to the re-ID gallery, if any
        # Ball-only (see CentroidTracker._update_ball / _ball_touches_a_player):
        # consecutive processed frames this "ball" has sat inside/right at a
        # player's own bounding box, whoever that happens to be. Tracks a
        # single moving person's slow drift indefinitely otherwise - a real
        # ball being dribbled still moves relative to a player's feet frame
        # to frame, this doesn't.
        self.ball_pinned_to_player_frames: int = 0

    def predicted_centroid(self) -> Tuple[float, float]:
        """
        Projects the track forward through however many frames it's been missing,
        using its last known velocity (damped so we don't overshoot on long gaps).
        This is what lets a player who's briefly occluded or missed by the detector
        get re-matched to their existing ID instead of spawning a new one.
        """
        steps = self.consecutive_lost + 1
        vx, vy = self.velocity
        damp = 0.85
        factor = sum(damp ** i for i in range(steps))
        return (self.centroid[0] + vx * factor, self.centroid[1] + vy * factor)

    def update(
        self,
        bbox: List[float],
        centroid: Tuple[float, float],
        frame_no: int,
        color_sample: Optional[Tuple[float, float, float]] = None,
        appearance_descriptor: Optional[np.ndarray] = None,
    ):
        prev_centroid = self.centroid
        dx = centroid[0] - prev_centroid[0]
        dy = centroid[1] - prev_centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)
        self.total_distance_px += dist

        frames_elapsed = max(1, frame_no - self.history[-1][0])
        alpha = 0.6
        inst_vx, inst_vy = dx / frames_elapsed, dy / frames_elapsed
        self.velocity = (
            alpha * inst_vx + (1 - alpha) * self.velocity[0],
            alpha * inst_vy + (1 - alpha) * self.velocity[1],
        )

        self.bbox = bbox
        self.centroid = centroid
        self.history.append((frame_no, centroid))
        self.consecutive_lost = 0
        self.lost_at_frame = None
        self.hits += 1
        if color_sample is not None:
            self.color_samples.append(color_sample)
        if appearance_descriptor is not None:
            if self.appearance_hist is None:
                self.appearance_hist = appearance_descriptor
            else:
                blended = 0.8 * self.appearance_hist + 0.2 * appearance_descriptor
                norm = np.linalg.norm(blended)
                self.appearance_hist = blended / norm if norm > 0 else blended

    def mark_lost(self):
        self.consecutive_lost += 1


class TeamAssignmentMixin:
    """
    HOME/AWAY jersey-color team assignment, shared by any tracker implementation
    that maintains a `self.tracks: Dict[int, TrackedObject]` of player tracks plus
    `self._team_reference_centroids` and `self._team_assign_counter` (a subclass's
    __init__ must set both - see CentroidTracker for the canonical example). Pulled
    out of CentroidTracker so a different tracking backend (e.g. BotSortPlayerTracker)
    can reuse the exact same, already-tuned color logic instead of duplicating it.
    """

    # A torso crop this dark (mean HSV "value" out of 255) carries essentially no
    # recoverable jersey color - just sensor noise - on real night-match footage
    # where half the pitch sits outside the floodlight pools. Found by directly
    # inspecting real torso crops from production footage: well-lit crops (clear,
    # identifiable jersey color) measured V 80-135; crops in shadow (indistinguishable
    # near-black torsos) measured V 43-80 with heavy overlap into the lit range, so a
    # threshold has to sit low enough to keep it from starving tracks of samples
    # entirely, but this one specific failure mode - averaging in near-black "colors"
    # from unlit frames - was confirmed to be *the* root cause of team misassignment
    # (both jersey colors and even the referee ending up in the same cluster).
    MIN_COLOR_SAMPLE_VALUE = 55.0

    def _sample_color(self, frame: Optional[np.ndarray], bbox: List[float]) -> Optional[Tuple[float, float, float]]:
        """
        Samples a player's torso region (avoids head/legs/background) and returns its
        mean (hue, saturation, value) - hue+saturation is what actually identifies a
        jersey *color*, independent of how brightly lit that patch of pitch happens to
        be; a raw mean BGR triplet conflates color with brightness, which is exactly
        what let two different jersey colors under different lighting collapse into
        the same cluster (see _assign_teams). Returns None for a crop too dark to
        carry reliable color information at all, rather than returning a near-black
        "color" that would just be noise once averaged into a track's color history.
        """
        if frame is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        torso_y1 = max(0, y1 + int(h * 0.15))
        torso_y2 = min(frame.shape[0], y1 + int(h * 0.55))
        x1c, x2c = max(0, x1), min(frame.shape[1], x2)
        if torso_y2 <= torso_y1 or x2c <= x1c:
            return None
        crop = frame[torso_y1:torso_y2, x1c:x2c]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
        if mean_hsv[2] < self.MIN_COLOR_SAMPLE_VALUE:
            return None
        return tuple(float(v) for v in mean_hsv)

    # How far (in normalized hue+saturation space) a track's color can sit from
    # its nearest team-cluster centroid before it's judged to not belong to
    # either team at all - i.e. a referee, in a third kit color distinct from
    # both teams. Calibrated directly from real footage: 11 confirmed real
    # players measured a max of 0.072 (median 0.044) from their own team's
    # centroid; a confirmed on-pitch referee (green kit, both teams white/navy)
    # measured 0.211 from his *nearest* centroid - a clean ~3x gap. 0.15 sits
    # in the middle of that gap with real margin on both sides.
    REFEREE_OUTLIER_DIST = 0.15

    def _assign_teams(self):
        """
        Assigns HOME / AWAY teams to player tracks by clustering jersey color (KMeans, k=2)
        on hue+saturation - the actual color-identity signal, independent of how brightly
        lit a given sample happened to be (see _sample_color; clustering on raw brightness-
        entangled color was confirmed to collapse two different jersey colors into the same
        cluster on real footage with uneven floodlighting). Position-based splitting breaks
        the moment players cross the pitch, so color is the primary signal; tracks without
        enough color samples yet fall back to a position split. A track's team is locked the
        first time it's assigned - re-clustering only fills in still-unlocked tracks from
        here on, it never revises an already-decided one (a real player doesn't switch sides
        mid-match; see TrackedObject.team_locked).

        A track whose color sits far from BOTH team clusters (see
        REFEREE_OUTLIER_DIST) is labeled "REFEREE" instead of forced into
        whichever team happens to be nominally closer - referees wear a kit
        distinct from both teams by design, so this is a real, usable signal,
        not a guess. Downstream event/stats code excludes REFEREE tracks from
        the analysis entirely (see track.py).
        """
        players = [t for t in self.tracks.values() if t.class_name == "player"]
        if len(players) < 2:
            return

        # Re-cluster periodically rather than every frame - jersey color is stable, so this
        # is both cheaper and avoids jittering labels on single noisy samples.
        self._team_assign_counter += 1
        with_color = [p for p in players if len(p.color_samples) >= 3]

        if len(with_color) >= 2 and self._team_assign_counter % 10 == 0:
            from scipy.cluster.vq import kmeans2

            # hue + saturation only (drop value/brightness) - two samples of the same
            # jersey under different lighting should cluster together, which a
            # brightness-sensitive feature (raw BGR, or HSV including V) won't do.
            mean_hsv = np.array([np.mean(p.color_samples[-20:], axis=0) for p in with_color])
            features = mean_hsv[:, :2]
            norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
            try:
                centroids, labels = kmeans2(norm, 2, seed=42, minit="++")

                # A single non-team kit color (one referee) barely nudges the two
                # team centroids - but real matches often have SEVERAL: both
                # teams' goalkeepers (each in their own third color, distinct from
                # their own outfield kit) plus a referee and assistant linesmen.
                # Each of those still gets forced onto whichever of the two
                # clusters its color happens to sit nominally closer to on this
                # first pass, which drags that cluster's centroid away from the
                # real jersey color by however much of the outlier leaked in -
                # and a dragged centroid then makes REFEREE_OUTLIER_DIST (measured
                # against a *clean* centroid) too lenient to catch the very
                # outliers that caused the drag. Refitting once, using only the
                # points that already look clean against the first-pass fit,
                # removes that contamination before the real HOME/AWAY/REFEREE
                # decision is made. Confirmed necessary on real professional-
                # match footage (Barcelona vs Espanyol) where two pink/orange
                # goalkeeper kits plus an orange-bibbed linesman were otherwise
                # split across both team buckets instead of excluded.
                dist_to_assigned = np.linalg.norm(norm - centroids[labels], axis=1)
                inlier_mask = dist_to_assigned <= self.REFEREE_OUTLIER_DIST
                if 2 <= int(inlier_mask.sum()) < len(norm):
                    try:
                        refit_centroids, _ = kmeans2(norm[inlier_mask], 2, seed=42, minit="++")
                        d_same = np.linalg.norm(refit_centroids[0] - centroids[0])
                        d_swap = np.linalg.norm(refit_centroids[0] - centroids[1])
                        centroids = refit_centroids[::-1] if d_swap < d_same else refit_centroids
                    except Exception:
                        pass  # keep the original (unrefined) centroids

                if self._team_reference_centroids is not None:
                    # Keep HOME/AWAY consistent across re-clusters (kmeans label order is arbitrary)
                    d_same = np.linalg.norm(centroids[0] - self._team_reference_centroids[0])
                    d_swap = np.linalg.norm(centroids[0] - self._team_reference_centroids[1])
                    if d_swap < d_same:
                        centroids = centroids[::-1]
                self._team_reference_centroids = centroids

                # Re-measure every point against the final (possibly refit)
                # centroids directly, rather than reusing the first-pass kmeans2
                # `labels` - those were assigned before contamination was removed
                # and can no longer be trusted for who's nearest to what.
                for p, feat in zip(with_color, norm):
                    if p.team_locked:
                        continue
                    dists = np.linalg.norm(centroids - feat, axis=1)
                    nearest = int(np.argmin(dists))
                    if dists[nearest] > self.REFEREE_OUTLIER_DIST:
                        p.team = "REFEREE"
                    else:
                        p.team = "HOME" if nearest == 0 else "AWAY"
                    p.team_locked = True
            except Exception:
                pass

        # Players without a stable color signal yet (new tracks) fall back to spatial split
        remaining = [p for p in players if not p.team_locked]
        if remaining:
            x_coords = [p.centroid[0] for p in players]
            median_x = np.median(x_coords)
            for p in remaining:
                p.team = "HOME" if p.centroid[0] < median_x else "AWAY"
                # Not locked yet - this is a low-confidence fallback (no color signal),
                # so let it keep being revised until real color samples accumulate and
                # the block above locks it in properly.


class CentroidTracker(TeamAssignmentMixin):
    """
    Multi-object tracker for football players & ball.
    Matches detections to existing tracks via optimal (Hungarian) assignment against
    motion-predicted positions, so brief misses/occlusions extend a track instead of
    killing it and spawning a new ID. Also clusters players into HOME/AWAY teams by
    jersey color rather than raw pitch position (which breaks once players cross over),
    and re-identifies players who reappear after a long gap (well past max_lost_frames)
    by matching a torso/leg color-histogram signature against a gallery of recently
    lost tracks, rather than spawning a brand new identity for them.
    """

    def __init__(
        self,
        max_lost_frames: int = 30,
        max_distance_px: float = 80.0,
        reid_gallery_max_age: int = 600,
        reid_max_distance: float = 0.3,
        ball_max_lost_frames: int = 300,
    ):
        self.max_lost_frames = max_lost_frames
        self.max_distance_px = max_distance_px
        self.reid_gallery_max_age = reid_gallery_max_age
        self.reid_max_distance = reid_max_distance
        # There's only ever one ball in play and real detections of it are sparse on
        # broadcast-distance footage - it needs a much longer leash than a player track
        # before being given up on, since a big gap can't possibly be a *different* ball.
        self.ball_max_lost_frames = ball_max_lost_frames
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.lost_gallery: Dict[int, TrackedObject] = {}
        self._team_reference_centroids: Optional[np.ndarray] = None
        self._team_assign_counter = 0
        # Rolling log of every raw "ball"-class candidate position seen recently
        # (frame_no, centroid), matched or not - see _is_static_ball_candidate.
        self._ball_candidate_log: List[Tuple[int, Tuple[float, float]]] = []

    def _get_centroid(self, bbox: List[float], class_name: str) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        if class_name == "player":
            # For players, bottom-center represents ground footprint on pitch
            return ((x1 + x2) / 2.0, y2)
        else:
            # For ball, geometric center
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _appearance_descriptor(self, frame: Optional[np.ndarray], bbox: List[float]) -> Optional[np.ndarray]:
        """
        Builds a re-ID signature from two color histograms (torso/shirt + legs/shorts+socks),
        since two regions capture more identity-specific signal than a single average color -
        e.g. two teammates in identical shirts often still differ in shorts/sock color, skin
        tone, or build. Cheap (no model, no training data) - meant to disambiguate a *specific*
        returning player from the gallery, not to substitute for real per-frame detection.
        """
        if frame is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        x1c, x2c = max(0, x1), min(frame.shape[1], x2)
        if x2c <= x1c or h <= 0:
            return None

        def region_hist(ya: int, yb: int) -> np.ndarray:
            ya, yb = max(0, ya), min(frame.shape[0], yb)
            if yb <= ya:
                return np.zeros(64, dtype=np.float32)
            crop = frame[ya:yb, x1c:x2c]
            if crop.size == 0:
                return np.zeros(64, dtype=np.float32)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            # L1 (sum-to-1), not L2 - _hist_distance's Bhattacharyya-coefficient
            # formula (sum of sqrt(a_i*b_i)) is only guaranteed to land in
            # [0, 1] when a and b are true probability distributions (L1-
            # normalized). Under L2 normalization a histogram compared to
            # ITSELF can score well above 1 (a busy real photo spreads mass
            # across many bins, and L2 keeps each bin's share larger than an
            # L1-normalized share would) - the formula then clips to a
            # distance of 0 regardless of what's actually being compared, so
            # every real (non-degenerate) crop silently reads as "identical"
            # to every other. That's a real bug, not just an approximation:
            # it means the re-ID gallery's appearance check was effectively a
            # no-op on real footage, leaving only the position gate to
            # disambiguate - exactly when two players cross paths or cluster
            # near the ball, which is when appearance is needed most.
            cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
            return hist.flatten()

        torso_hist = region_hist(y1 + int(h * 0.15), y1 + int(h * 0.55))
        legs_hist = region_hist(y1 + int(h * 0.55), y1 + int(h * 0.95))
        descriptor = np.concatenate([torso_hist, legs_hist]).astype(np.float32)
        # L1-normalize the combined vector too, so it's one valid probability
        # distribution over both regions together (each region individually
        # already sums to 1, so the raw concatenation sums to 2).
        total = descriptor.sum()
        return descriptor / total if total > 0 else descriptor

    def _hist_distance(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Hellinger distance between two normalized histograms: 0 = identical, 1 = disjoint."""
        if a is None or b is None:
            return float("inf")
        bc = np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)))
        return float(np.sqrt(max(0.0, 1.0 - bc)))

    # Rough cap on how far a player can plausibly have moved since being lost,
    # in pixels/frame-unit. Measured actual player motion in this footage
    # (frame-to-frame centroid deltas): median 4.45, p90 13.46, p99 28.17
    # px/frame-unit. The budget needs real margin above that - the "predicted"
    # position is a straight-line extrapolation of last known velocity, which
    # diverges fast from a player who changes direction, slows, or stops during
    # the miss - so this is set well above the measured p99 rather than close
    # to it; a tight budget mostly rejects correct matches (and previously
    # made fragmentation worse, not better - see git history), while the
    # appearance histogram is what actually discriminates between candidates.
    REID_MAX_PX_PER_FRAME = 60.0
    REID_MAX_PX_ABSOLUTE = 1200.0

    # A track needs at least one real second sighting (not just its initial
    # spawn) before it's allowed into the re-ID gallery. Found on production
    # footage: a single-frame detector false positive (a bright line/reflection
    # misread as a "player") spawned a track that never matched again - but
    # still sat in the gallery for hundreds of frames, well within
    # REID_MAX_PX_ABSOLUTE's reach, until it was "found" and merged with a
    # completely unrelated real player's trajectory later in the match. A real
    # player is matched again within a frame or two almost always; something
    # that never gets a second hit before expiring is far more likely a one-off
    # artifact than a real person who happened to vanish instantly, so it isn't
    # worth keeping around as re-ID bait.
    CONFIRM_HITS = 2

    def _try_reid_match(
        self, descriptor: Optional[np.ndarray], centroid: Tuple[float, float], frame_no: int
    ) -> Optional[TrackedObject]:
        """
        Looks for a lost player whose appearance most closely matches a new detection.
        Appearance (jersey-color histogram) alone can't disambiguate teammates - two
        players on the same side look near-identical by color, so pure appearance
        matching was misassigning a reappearing player to whichever lost teammate's
        histogram happened to be closest, or missing the match and spawning a new
        track ID (the main driver of track-count fragmentation). Gating candidates by
        plausible position first - projecting each lost track's last known motion
        forward through however many frames it's been missing - means only players
        who could actually be at this spot are even considered before appearance
        breaks the tie.
        """
        if descriptor is None or not self.lost_gallery:
            return None
        best_id, best_dist = None, self.reid_max_distance
        for t_id, t in self.lost_gallery.items():
            elapsed = max(1, frame_no - (t.lost_at_frame or frame_no))
            damp = 0.85
            factor = sum(damp ** i for i in range(elapsed))
            predicted = (t.centroid[0] + t.velocity[0] * factor, t.centroid[1] + t.velocity[1] * factor)
            max_dist_px = min(self.REID_MAX_PX_ABSOLUTE, self.REID_MAX_PX_PER_FRAME * elapsed)
            if math.hypot(centroid[0] - predicted[0], centroid[1] - predicted[1]) > max_dist_px:
                continue
            dist = self._hist_distance(descriptor, t.appearance_hist)
            if dist < best_dist:
                best_dist = dist
                best_id = t_id
        return self.lost_gallery.pop(best_id) if best_id is not None else None

    def update(self, detections: List[DetectionResult], frame_no: int, frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """
        Update tracker with detections from the current frame.
        `frame` (optional) is the raw BGR image, used only to sample jersey color for team assignment.
        Returns list of active TrackedObjects for this frame.
        """
        # Separate player detections from ball
        player_dets = [d for d in detections if d.class_name == "player"]
        ball_dets = [d for d in detections if d.class_name == "ball"]

        active_tracks: List[TrackedObject] = []

        # 1. Match player detections to existing player tracks
        player_tracks = {t_id: t for t_id, t in self.tracks.items() if t.class_name == "player"}
        matched_tracks = self._match_detections(player_tracks, player_dets, frame_no, frame=frame, try_reid=True)
        active_tracks.extend(matched_tracks)

        # 2. Match or track the ball - no distance/identity gating (see _update_ball)
        active_tracks.extend(self._update_ball(ball_dets, frame_no, player_dets=player_dets))

        # 3. Move stale player tracks into the re-ID gallery instead of deleting them outright,
        # so a player who reappears after a longer gap (occlusion, camera pan, brief exit) can
        # be matched back to their original identity by appearance rather than spawning a new one.
        expired_ids = [
            t_id for t_id, t in self.tracks.items()
            if t.consecutive_lost > (self.ball_max_lost_frames if t.class_name == "ball" else self.max_lost_frames)
        ]
        for t_id in expired_ids:
            t = self.tracks.pop(t_id)
            if t.class_name == "player" and t.appearance_hist is not None and t.hits >= self.CONFIRM_HITS:
                t.lost_at_frame = frame_no
                self.lost_gallery[t_id] = t
            # else: never confirmed by a second real sighting - dropped outright
            # rather than kept as re-ID bait (see CONFIRM_HITS).

        # 3b. Permanently drop gallery entries that have been unmatched for too long
        stale_gallery_ids = [
            t_id for t_id, t in self.lost_gallery.items()
            if (frame_no - (t.lost_at_frame or frame_no)) > self.reid_gallery_max_age
        ]
        for t_id in stale_gallery_ids:
            del self.lost_gallery[t_id]

        # 4. Assign teams based on jersey color (falls back to spatial split if no color signal yet)
        self._assign_teams()

        return active_tracks

    # How long a rolling window (in raw video frame_no units, not processed-frame
    # calls - stays meaningful regardless of sampled_rate) to remember every raw
    # "ball"-class candidate position, and how many separate sightings at
    # essentially the same spot within that window mark it as a static false
    # positive rather than the real ball. Calibrated from a real, confirmed case:
    # a sponsor-board logo got matched as "the ball" for 2.5+ seconds spanning a
    # free-kick strike, because it's a static, always-visible, confidently-
    # detected shape - exactly what a real ball moving fast enough to blur or
    # briefly leave frame cannot be. 90 frames is ~3.75s at a typical 24fps.
    STATIC_BALL_CANDIDATE_WINDOW_FRAMES = 90
    STATIC_BALL_CANDIDATE_MIN_SIGHTINGS = 3
    STATIC_BALL_CANDIDATE_TOLERANCE_PX = 15.0

    def _is_static_ball_candidate(self, pos: Tuple[float, float], frame_no: int) -> bool:
        """True if `pos` has shown up as a "ball" candidate suspiciously often within
        the recent window - a real ball in flight can't produce that; a fixed
        background object confidently (mis)detected as a ball repeatedly can."""
        sightings = sum(
            1 for f, p in self._ball_candidate_log
            if frame_no - f <= self.STATIC_BALL_CANDIDATE_WINDOW_FRAMES
            and math.hypot(pos[0] - p[0], pos[1] - p[1]) <= self.STATIC_BALL_CANDIDATE_TOLERANCE_PX
        )
        return sightings >= self.STATIC_BALL_CANDIDATE_MIN_SIGHTINGS

    # How many consecutive processed frames a "ball" can appear to sit inside
    # (or right at the edge of) the same nearby player's own bounding box
    # before we get suspicious it isn't really the ball at all - just a small
    # mis-detected part of that one player (a shoe, sock, or reflective badge)
    # that happens to satisfy ordinary trajectory continuity every frame,
    # since it moves exactly however that person moves. _is_static_ball_
    # candidate can't catch this: it only ever runs while *recovering* from a
    # loss, but a false lock like this is never "lost" (it's continuously,
    # confidently re-matched every frame), so it can persist indefinitely.
    # Confirmed on real footage: a slow-walking match official kept a "ball"
    # track pinned to himself for 10+ continuous seconds this way. A real
    # ball at a player's feet (dribbling, shielding) still drifts relative to
    # them frame to frame; this catches only the case where it doesn't.
    # ~2s at a typical 25fps.
    BALL_PINNED_TO_PLAYER_MAX_FRAMES = 50
    # Padding (px) around a player's box that still counts as "at his feet" -
    # a real ball there sits at/just past the box's bottom edge, not
    # necessarily strictly inside it.
    BALL_PINNED_TO_PLAYER_PAD_PX = 10.0

    def _ball_touches_a_player(
        self, centroid: Tuple[float, float], player_dets: List[DetectionResult]
    ) -> bool:
        cx, cy = centroid
        pad = self.BALL_PINNED_TO_PLAYER_PAD_PX
        for d in player_dets:
            x1, y1, x2, y2 = d.bbox
            if x1 - pad <= cx <= x2 + pad and y1 - pad <= cy <= y2 + pad:
                return True
        return False

    def _update_ball(
        self,
        ball_dets: List[DetectionResult],
        frame_no: int,
        player_dets: Optional[List[DetectionResult]] = None,
    ) -> List[TrackedObject]:
        """
        Unlike players, there's only ever one real ball in play - any new ball detection
        almost certainly IS the existing ball track, however far it's moved or however
        long the gap since the last sighting (ball detections are sparse on real footage;
        see YOLOv8Detector.detect_ball_tiled). So this skips distance-gated matching
        entirely rather than risk spawning a new ID every time the gap exceeds a fixed
        threshold - there's no second ball it could be confused with.

        `player_dets` (this frame's player boxes), if given, guards against the
        failure mode _is_static_ball_candidate can't: see
        BALL_PINNED_TO_PLAYER_MAX_FRAMES.
        """
        player_dets = player_dets or []
        # Logged before anything else - even a candidate rejected below still
        # counts as "this position keeps showing up", which is exactly the
        # signal _is_static_ball_candidate needs.
        for d in ball_dets:
            self._ball_candidate_log.append((frame_no, self._get_centroid(d.bbox, "ball")))
        self._ball_candidate_log = [
            (f, p) for f, p in self._ball_candidate_log
            if frame_no - f <= self.STATIC_BALL_CANDIDATE_WINDOW_FRAMES
        ]

        ball_tracks = {t_id: t for t_id, t in self.tracks.items() if t.class_name == "ball"}

        if not ball_dets:
            for t in ball_tracks.values():
                t.mark_lost()
            return []

        if ball_tracks:
            t = next(iter(ball_tracks.values()))
            # Only accept a detection consistent with the ball's own trajectory -
            # a real ball can't teleport across the pitch in a single processed
            # frame. Budget scales with how long it's been missing (a real gap
            # does allow real movement), so this self-widens to accept any
            # detection once the gap is long enough to plausibly be anywhere -
            # no separate "just trust the best one" fallback needed, since that
            # fallback is exactly what let a same-frame false positive (crowd,
            # advertising board, floodlight) hijack the track every time real
            # ball recall missed the actual ball that frame.
            predicted = t.predicted_centroid()
            # Capped rather than growing forever: a genuinely long absence (many
            # seconds) should recover via full track expiry + a fresh trusted
            # first-sighting (see below), not by this same track's jump budget
            # eventually becoming "anywhere in the frame is plausible."
            max_jump = min(700.0, 140.0 * (t.consecutive_lost + 1))
            candidates = [
                d for d in ball_dets
                if math.hypot(*(np.subtract(self._get_centroid(d.bbox, "ball"), predicted))) <= max_jump
            ]
            if t.consecutive_lost > 0:
                # Only relevant while *recovering* from a loss - a ball that's
                # been continuously, confidently tracked frame-to-frame is
                # trusted even if genuinely stationary (a real dead-ball
                # moment). The risk is specifically re-acquiring onto a static
                # decoy after real detection dropped out.
                candidates = [
                    d for d in candidates
                    if not self._is_static_ball_candidate(self._get_centroid(d.bbox, "ball"), frame_no)
                ]
            if not candidates:
                t.mark_lost()
                return []

            best_det = max(candidates, key=lambda d: d.confidence)
            centroid = self._get_centroid(best_det.bbox, "ball")

            if self._ball_touches_a_player(centroid, player_dets):
                t.ball_pinned_to_player_frames += 1
            else:
                t.ball_pinned_to_player_frames = 0

            if t.ball_pinned_to_player_frames > self.BALL_PINNED_TO_PLAYER_MAX_FRAMES:
                # Likely locked onto a person, not the ball (see
                # BALL_PINNED_TO_PLAYER_MAX_FRAMES) - drop this track outright
                # rather than keep extending it, and try a fresh acquisition
                # this same frame preferring a candidate that ISN'T pinned to
                # a player, if one's available.
                del self.tracks[t.track_id]
                non_player_dets = [
                    d for d in ball_dets
                    if not self._ball_touches_a_player(self._get_centroid(d.bbox, "ball"), player_dets)
                ]
                return self._acquire_fresh_ball(non_player_dets or ball_dets, frame_no)

            t.update(best_det.bbox, centroid, frame_no)
            return [t]

        # First sighting (or the previous track already expired) - no trajectory to
        # judge candidates against yet, so just trust the most confident detection,
        # excluding known-static ones for the same reason as the recovery path above.
        return self._acquire_fresh_ball(ball_dets, frame_no)

    def _acquire_fresh_ball(
        self, ball_dets: List[DetectionResult], frame_no: int
    ) -> List[TrackedObject]:
        """Starts a brand new ball track from whichever of `ball_dets` looks most
        confident, excluding known-static decoys (see _is_static_ball_candidate).
        Shared by the natural first-sighting path and by _update_ball's forced
        re-acquisition after dropping a track pinned to a player."""
        fresh_candidates = [
            d for d in ball_dets
            if not self._is_static_ball_candidate(self._get_centroid(d.bbox, "ball"), frame_no)
        ]
        if not fresh_candidates:
            return []
        best_det = max(fresh_candidates, key=lambda d: d.confidence)
        centroid = self._get_centroid(best_det.bbox, "ball")

        track = TrackedObject(
            track_id=self.next_track_id,
            class_name="ball",
            bbox=best_det.bbox,
            centroid=centroid,
            frame_no=frame_no,
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return [track]

    def _match_detections(
        self,
        existing_tracks: Dict[int, TrackedObject],
        detections: List[DetectionResult],
        frame_no: int,
        max_dist: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        try_reid: bool = False,
    ) -> List[TrackedObject]:
        max_distance = max_dist if max_dist is not None else self.max_distance_px
        updated_tracks: List[TrackedObject] = []

        def spawn_or_revive(det: DetectionResult, centroid: Tuple[float, float]) -> TrackedObject:
            color = self._sample_color(frame, det.bbox) if det.class_name == "player" else None
            descriptor = self._appearance_descriptor(frame, det.bbox) if try_reid else None
            revived = self._try_reid_match(descriptor, centroid, frame_no) if try_reid else None
            if revived is not None:
                revived.update(det.bbox, centroid, frame_no, color_sample=color, appearance_descriptor=descriptor)
                self.tracks[revived.track_id] = revived
                return revived

            track = TrackedObject(
                track_id=self.next_track_id,
                class_name=det.class_name,
                bbox=det.bbox,
                centroid=centroid,
                frame_no=frame_no,
            )
            if color is not None:
                track.color_samples.append(color)
            track.appearance_hist = descriptor
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
            return track

        if len(existing_tracks) == 0:
            for det in detections:
                centroid = self._get_centroid(det.bbox, det.class_name)
                updated_tracks.append(spawn_or_revive(det, centroid))
            return updated_tracks

        track_ids = list(existing_tracks.keys())
        # Match against each track's *predicted* position, not its last known one, so a track
        # that's been missing for a few frames (occlusion, dropped detection) is still findable.
        predicted_centroids = [existing_tracks[t_id].predicted_centroid() for t_id in track_ids]

        det_centroids = [self._get_centroid(d.bbox, d.class_name) for d in detections]

        if len(det_centroids) == 0:
            for t_id in track_ids:
                existing_tracks[t_id].mark_lost()
            return updated_tracks

        # Cost matrix (Euclidean distance) + optimal (Hungarian) assignment instead of greedy
        # nearest-neighbor, which avoids incorrect swaps when players are close together.
        cost_matrix = np.zeros((len(track_ids), len(det_centroids)), dtype=float)
        for i, tc in enumerate(predicted_centroids):
            for j, dc in enumerate(det_centroids):
                cost_matrix[i, j] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

        # Gate implausible pairs to a large sentinel *before* solving, not just
        # filtered out after. Hungarian minimizes the GLOBAL sum, so a single
        # isolated track with no real candidate nearby (e.g. a player far from
        # everyone else, genuinely occluded/missed that frame) would otherwise
        # get assigned whatever detection is "least bad" for it - and since that
        # reassignment can ripple through the whole matrix, it can bump two
        # completely unrelated, genuinely-adjacent tracks off their own excellent
        # (sub-30px) matches onto equally-implausible ones, so all three end up
        # rejected and re-spawned as brand new IDs instead of just the one that
        # actually had no match. A sentinel cost (versus real, always-smaller
        # distances) can never be worth "trading into" a nearby track's good
        # match, so it can't contaminate assignments outside its own row/column.
        gated_cost_matrix = np.where(cost_matrix > max_distance, 1e6, cost_matrix)
        row_ind, col_ind = linear_sum_assignment(gated_cost_matrix)

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > max_distance:
                continue

            t_id = track_ids[r]
            det = detections[c]
            centroid = det_centroids[c]

            color = self._sample_color(frame, det.bbox) if det.class_name == "player" else None
            descriptor = self._appearance_descriptor(frame, det.bbox) if try_reid else None
            existing_tracks[t_id].update(det.bbox, centroid, frame_no, color_sample=color, appearance_descriptor=descriptor)
            updated_tracks.append(existing_tracks[t_id])

            assigned_rows.add(r)
            assigned_cols.add(c)

        # Mark unmatched existing tracks as lost
        for i, t_id in enumerate(track_ids):
            if i not in assigned_rows:
                existing_tracks[t_id].mark_lost()

        # Unmatched detections: try to re-identify against recently-lost tracks before
        # minting a brand new ID
        for j, det in enumerate(detections):
            if j not in assigned_cols:
                centroid = det_centroids[j]
                updated_tracks.append(spawn_or_revive(det, centroid))

        return updated_tracks

