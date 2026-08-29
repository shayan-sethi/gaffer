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


class CentroidTracker:
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

    def _get_centroid(self, bbox: List[float], class_name: str) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        if class_name == "player":
            # For players, bottom-center represents ground footprint on pitch
            return ((x1 + x2) / 2.0, y2)
        else:
            # For ball, geometric center
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _sample_color(self, frame: Optional[np.ndarray], bbox: List[float]) -> Optional[Tuple[float, float, float]]:
        """Samples the mean BGR color of a player's torso region (avoids head/legs/background)."""
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
        mean_color = crop.reshape(-1, 3).mean(axis=0)
        return tuple(float(v) for v in mean_color)

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
            cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
            return hist.flatten()

        torso_hist = region_hist(y1 + int(h * 0.15), y1 + int(h * 0.55))
        legs_hist = region_hist(y1 + int(h * 0.55), y1 + int(h * 0.95))
        descriptor = np.concatenate([torso_hist, legs_hist]).astype(np.float32)
        norm = np.linalg.norm(descriptor)
        return descriptor / norm if norm > 0 else descriptor

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
        active_tracks.extend(self._update_ball(ball_dets, frame_no))

        # 3. Move stale player tracks into the re-ID gallery instead of deleting them outright,
        # so a player who reappears after a longer gap (occlusion, camera pan, brief exit) can
        # be matched back to their original identity by appearance rather than spawning a new one.
        expired_ids = [
            t_id for t_id, t in self.tracks.items()
            if t.consecutive_lost > (self.ball_max_lost_frames if t.class_name == "ball" else self.max_lost_frames)
        ]
        for t_id in expired_ids:
            t = self.tracks.pop(t_id)
            if t.class_name == "player" and t.appearance_hist is not None:
                t.lost_at_frame = frame_no
                self.lost_gallery[t_id] = t

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

    def _update_ball(self, ball_dets: List[DetectionResult], frame_no: int) -> List[TrackedObject]:
        """
        Unlike players, there's only ever one real ball in play - any new ball detection
        almost certainly IS the existing ball track, however far it's moved or however
        long the gap since the last sighting (ball detections are sparse on real footage;
        see YOLOv8Detector.detect_ball_tiled). So this skips distance-gated matching
        entirely rather than risk spawning a new ID every time the gap exceeds a fixed
        threshold - there's no second ball it could be confused with.
        """
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
            if not candidates:
                t.mark_lost()
                return []

            best_det = max(candidates, key=lambda d: d.confidence)
            centroid = self._get_centroid(best_det.bbox, "ball")
            t.update(best_det.bbox, centroid, frame_no)
            return [t]

        # First sighting (or the previous track already expired) - no trajectory to
        # judge candidates against yet, so just trust the most confident detection.
        best_det = max(ball_dets, key=lambda d: d.confidence)
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

    def _assign_teams(self):
        """
        Assigns HOME / AWAY teams to player tracks by clustering jersey color (KMeans, k=2).
        Position-based splitting breaks the moment players cross the pitch, so color is the
        primary signal; tracks without enough color samples yet fall back to a position split.
        A track's team is locked the first time it's assigned - re-clustering only fills in
        still-unlocked tracks from here on, it never revises an already-decided one (a real
        player doesn't change sides mid-match; see TrackedObject.team_locked).
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

            features = np.array([np.mean(p.color_samples[-20:], axis=0) for p in with_color])
            norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
            try:
                centroids, labels = kmeans2(norm, 2, seed=42, minit="++")
                if self._team_reference_centroids is not None:
                    # Keep HOME/AWAY consistent across re-clusters (kmeans label order is arbitrary)
                    d_same = np.linalg.norm(centroids[0] - self._team_reference_centroids[0])
                    d_swap = np.linalg.norm(centroids[0] - self._team_reference_centroids[1])
                    if d_swap < d_same:
                        centroids = centroids[::-1]
                        labels = 1 - labels
                self._team_reference_centroids = centroids
                for p, label in zip(with_color, labels):
                    if p.team_locked:
                        continue
                    p.team = "HOME" if label == 0 else "AWAY"
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
