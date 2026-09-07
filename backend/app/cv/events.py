import math
from typing import List, Dict, Any, Optional, Tuple
from app.cv.tracker import TrackedObject
from app.cv.calibration import PitchCalibration


class DetectedEvent:
    def __init__(
        self,
        frame_no: int,
        timestamp_ms: int,
        event_type: str,
        player_track_id: Optional[int] = None,
        team: Optional[str] = None,
        data_json: Optional[Dict[str, Any]] = None,
    ):
        self.frame_no = frame_no
        self.timestamp_ms = timestamp_ms
        self.event_type = event_type
        self.player_track_id = player_track_id
        self.team = team
        self.data_json = data_json or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_no": self.frame_no,
            "timestamp_ms": self.timestamp_ms,
            "event_type": self.event_type,
            "player_track_id": self.player_track_id,
            "team": self.team,
            "data_json": self.data_json,
        }


class EventDetector:
    """
    Rule-based event detection engine analyzing frame-by-frame tracked objects.
    Detects:
    - GOAL: Ball enters left or right goal zone
    - SHOT: Ball accelerated towards goal from player
    - PASS: Ball transitions from one player's possession to a teammate
    - POSSESSION_CHANGE: Ball transitions to opposing team
    """

    # Shot detection / xG tuning.
    SHOT_SPEED_KMH = 55.0  # comfortably above typical pass/dribble/clearance speed
    # No real football shot exceeds this by any meaningful margin (the fastest
    # officially recorded shot is ~131 km/h, by a professional) - found on
    # production footage that most computed "shots" were 140-165 km/h, an order
    # of magnitude beyond anything a real strike produces. That's ball-tracking
    # noise (the single "ball" track briefly snapping onto a spurious detection
    # elsewhere in frame - a false positive, or a second real ball/object -
    # rather than following the real ball), not a fast shot, and needs an upper
    # bound the same way player speed already has one (see StatsCalculator's
    # 35 km/h filter) - SHOT_SPEED_KMH alone only sets a floor.
    SHOT_MAX_SPEED_KMH = 120.0
    SHOT_MAX_DIST_M = 30.0  # only count as a shot if launched from within this
    # many real-world meters of the target goal - filters out hopeful long
    # balls/clearances upfield that happen to be fast and vaguely goal-ward.
    # A FIXED distance, not a fraction of pitch_length_m: realistic shooting
    # range is a human/technique constant, not a percentage of the pitch -
    # expressing it as a fraction (an earlier version used 0.4 * pitch_length_m)
    # meant ~16m on the 40m turf pitch this was tuned against, but silently
    # became ~42m - nearly half the pitch - on a real 105m 11-a-side pitch,
    # confirmed on real footage: every flagged "shot" clustered at 22-42m with
    # none closer, and a manual video check at 33m showed calm, uncontested
    # midfield build-up play, nowhere near a shooting position.
    SHOT_COOLDOWN_SEC = 3.0
    SHOT_MIN_COS_SIM = 0.8  # velocity must point within ~37 degrees of the goal
    # (raised from 0.6/~53 degrees - too loose, let diagonal clearances and
    # cross-field balls that were only vaguely goal-ward count as shots.)
    # Ball velocity is measured over at least this much elapsed time, not
    # frame-to-frame - process_frame runs on every single video frame (1/fps
    # seconds apart) since interpolation fills every frame, so a frame-to-frame
    # velocity divides ordinary few-pixel detection jitter by a tiny time step
    # and produces nonsense (1000+ km/h "shots"). A real shot's speed is still
    # obvious over a third-of-a-second window; jitter isn't.
    SHOT_MIN_WINDOW_SEC = 0.3
    SHOT_MAX_WINDOW_SEC = 1.0

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: float = 30.0,
        pitch_bounds: Optional[Tuple[int, int, int, int]] = None,
        pitch_length_m: float = 40.0,
        pitch_width_m: float = 20.0,
        calibration: Optional[PitchCalibration] = None,
        home_attacks_right: bool = True,
        goal_axis: str = "x",
    ):
        self.width = frame_width
        self.height = frame_height
        self.fps = fps
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        # Which goal each team is trying to score in - purely a camera/kickoff
        # convention for this specific recording, not a universal rule, so it
        # must be passed in rather than assumed. Getting it backwards silently
        # swaps GOAL/SHOT team attribution, xG target, and attacking-third
        # zone stats for the whole match - it does NOT just affect corners.
        self.home_attacks_right = home_attacks_right
        # Which pixel axis the two goals sit along, for the UNCALIBRATED
        # fallback only (a real calibration already handles either camera
        # orientation transparently, since it always maps into the same
        # canonical world frame regardless of pixel layout - see
        # update_pitch_bounds). "x" is a sideline camera (goals at left/right
        # pixel edges, the original/default assumption). "y" is a camera that
        # shoots down the pitch length from behind one goal (e.g. mounted
        # above a goal) - the goals sit at the top/bottom pixel edges instead,
        # and treating them as left/right would place goal zones at the
        # sideline hoardings, not the actual goals.
        if goal_axis not in ("x", "y"):
            raise ValueError(f"goal_axis must be 'x' or 'y', got {goal_axis!r}")
        self.goal_axis = goal_axis
        # See PitchCalibration - when set, shot distance/speed/angle are computed
        # via the real per-camera homography instead of the linear px_to_m
        # fallback below.
        self.calibration = calibration

        # Goal zones sit just inside where the pitch actually starts/ends in frame,
        # not the raw frame edges - see update_pitch_bounds for the full rationale
        # and why this has to be re-derivable, not computed once and locked in.
        self.update_pitch_bounds(pitch_bounds)

        # Per-team touch counts by pitch third, relative to which goal each team
        # attacks (see _zone_for_team, which uses the same home_attacks_right
        # convention as the GOAL check below). This only needs ball position +
        # current possessor team, both already tracked - no new detection
        # capability required.
        self.zone_touches: Dict[str, Dict[str, int]] = {
            "HOME": {"DEFENSIVE": 0, "MIDDLE": 0, "ATTACKING": 0},
            "AWAY": {"DEFENSIVE": 0, "MIDDLE": 0, "ATTACKING": 0},
        }

        self.last_possessor_id: Optional[int] = None
        self.last_possessor_team: Optional[str] = None
        # Frame counts a team spent in current possession (ball within
        # possession_radius of one of their players) - a direct measure of
        # actual time-with-the-ball, used for possession% instead of the
        # previous pass-count ratio (a noisy proxy: a team can hold the ball
        # for a long unbroken spell without ever "passing" in open play, or
        # complete several short passes while barely holding it overall).
        self.possession_frames: Dict[str, int] = {"HOME": 0, "AWAY": 0}
        # Same idea per-player, for each player's individual possession% (replacing
        # the previous per-player proxy of "share of events attributed to them").
        self.player_possession_frames: Dict[int, int] = {}
        self.last_goal_frame: int = -300  # Cooldown between goal events (10 sec @ 30fps)
        # A stray single-frame ball misdetection landing in the goal zone shouldn't
        # count as a goal - require the ball to actually linger there for a couple of
        # processed frames, same as a real ball settling in the net would.
        self._goal_zone_side: Optional[str] = None
        self._goal_zone_streak: int = 0
        self._goal_zone_required_streak: int = 2

        self._last_ball_pos: Optional[Tuple[float, float]] = None
        self._last_ball_frame: Optional[int] = None
        self.last_shot_frame: int = -int(fps * self.SHOT_COOLDOWN_SEC)

        self.events: List[DetectedEvent] = []

    def update_pitch_bounds(self, pitch_bounds: Optional[Tuple[int, int, int, int]]) -> None:
        """
        (Re)derives goal zones, pitch extent, and each team's target-goal
        centroid from a pitch bounding box. Callable after construction, not
        just once in __init__ - Gaffer's camera pans/zooms to follow play, so
        a goal zone computed from wherever the pitch happened to sit in ONE
        frame drifts wrong for the rest of the match: on real footage this is
        exactly what let two straight-line ball interpolations pass through a
        stale goal-zone box and fabricate GOAL events, while a real goal (the
        camera having panned since) never fell inside it at all. Callers that
        can supply a per-frame or periodically-refreshed pitch_bounds (see
        detect_events_task's pitch_bounds_history) should call this before
        each process_frame instead of relying solely on the constructor value.
        """
        # Goal zones sit just inside where the pitch actually starts/ends in frame,
        # not the raw frame edges - a fixed/wide camera usually shows crowd, boards,
        # or run-off space beyond the touchline, so "left/right 8% of the frame" can
        # land entirely off the pitch. Falls back to the frame-edge heuristic only
        # when pitch segmentation wasn't available (e.g. the synthetic demo canvas).
        if pitch_bounds is not None:
            px1, py1, px2, py2 = pitch_bounds
            pitch_w = max(1, px2 - px1)
            pitch_h = max(1, py2 - py1)
            if self.goal_axis == "x":
                margin = pitch_w * 0.06
                # A goal only occupies the middle band of the touchline edge -
                # without this, a corner-kick area (same x-band, top/bottom of
                # the pitch) would also register as "the goal".
                y_top, y_bot = py1 + pitch_h * 0.35, py1 + pitch_h * 0.65
                self.left_goal = (px1, y_top, px1 + margin, y_bot)
                self.right_goal = (px2 - margin, y_top, px2, y_bot)
            else:
                margin = pitch_h * 0.06
                x_lo, x_hi = px1 + pitch_w * 0.35, px1 + pitch_w * 0.65
                self.left_goal = (x_lo, py1, x_hi, py1 + margin)  # far end (low y)
                self.right_goal = (x_lo, py2 - margin, x_hi, py2)  # near end (high y)
            self.pitch_x1, self.pitch_x2 = px1, px2
            self.pitch_y1, self.pitch_y2 = py1, py2
        elif self.goal_axis == "x":
            self.left_goal = (0, self.height * 0.35, self.width * 0.08, self.height * 0.65)
            self.right_goal = (self.width * 0.92, self.height * 0.35, self.width, self.height * 0.65)
            self.pitch_x1, self.pitch_x2 = 0, self.width
            self.pitch_y1, self.pitch_y2 = 0, self.height
        else:
            self.left_goal = (self.width * 0.35, 0, self.width * 0.65, self.height * 0.08)
            self.right_goal = (self.width * 0.35, self.height * 0.92, self.width * 0.65, self.height)
            self.pitch_x1, self.pitch_x2 = 0, self.width
            self.pitch_y1, self.pitch_y2 = 0, self.height

        if self.calibration is not None:
            # Overrides the pitch_bounds/frame-edge heuristic above with the real
            # calibrated goal-line position, projected into pixel space. That
            # heuristic assumes a sideline camera where the goals sit at the
            # left/right edges of the pitch bounding box - wrong for a camera
            # that shoots down the length of the pitch, where a goal can appear
            # well inside the frame (e.g. near the top). The real goal line sits
            # at world x=0 / x=pitch_length_m by construction (see
            # PitchCalibration's TL/TR/BR/BL corner convention), so project that
            # rectangle forward through the calibration instead of guessing from
            # pixel geometry.
            def _project_world_box(x1_m: float, x2_m: float, y1_m: float, y2_m: float):
                xs, ys = zip(
                    self.calibration.world_to_pixel(x1_m, y1_m),
                    self.calibration.world_to_pixel(x2_m, y1_m),
                    self.calibration.world_to_pixel(x2_m, y2_m),
                    self.calibration.world_to_pixel(x1_m, y2_m),
                )
                # The rectangle's corners don't stay axis-aligned under a
                # perspective transform, but the box is small (a goal mouth, not
                # the whole pitch), so its bounding box is a fine approximation
                # for a goal-entry test.
                return (min(xs), min(ys), max(xs), max(ys))

            goal_depth_m = self.pitch_length_m * 0.06
            y_lo_m, y_hi_m = self.pitch_width_m * 0.35, self.pitch_width_m * 0.65
            self.left_goal = _project_world_box(0.0, goal_depth_m, y_lo_m, y_hi_m)
            self.right_goal = _project_world_box(
                self.pitch_length_m - goal_depth_m, self.pitch_length_m, y_lo_m, y_hi_m
            )

        # Pixel -> real-world scale (same convention as StatsCalculator) so shot
        # speed/distance thresholds are expressed in km/h and meters, not pixels.
        # Which world dimension (pitch length vs width) each PIXEL axis scales
        # follows goal_axis - on an "x" camera pixel-x runs the pitch's length,
        # but on a "y" camera (goals at top/bottom) it's pixel-y that does, and
        # getting this backwards would scale along-pitch motion by the width
        # ratio (or vice versa), corrupting every shot speed/distance/angle.
        if self.goal_axis == "x":
            self.px_to_m_x = self.pitch_length_m / max(1, self.pitch_x2 - self.pitch_x1)
            self.px_to_m_y = self.pitch_width_m / max(1, self.pitch_y2 - self.pitch_y1)
        else:
            self.px_to_m_x = self.pitch_width_m / max(1, self.pitch_x2 - self.pitch_x1)
            self.px_to_m_y = self.pitch_length_m / max(1, self.pitch_y2 - self.pitch_y1)

        def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # Each team's *target* (the goal they're shooting at) is whichever goal
        # home_attacks_right says they attack - so each other's.
        self.home_target_goal = _center(self.right_goal if self.home_attacks_right else self.left_goal)
        self.away_target_goal = _center(self.left_goal if self.home_attacks_right else self.right_goal)

    def _to_world_m(self, x: float, y: float) -> Tuple[float, float]:
        """
        Pixel -> real-world pitch meters, via the real per-camera homography if
        calibrated, otherwise the linear px_to_m_x/y approximation (offset from
        the pitch's top-left so that two such points still difference out to the
        same delta the old code computed directly on raw pixels). The returned
        tuple is always (along-pitch-length, across-pitch-width) regardless of
        goal_axis - callers (shot direction/angle, zone thirds) rely on that
        canonical order, not on which raw pixel axis it came from.
        """
        if self.calibration is not None:
            return self.calibration.pixel_to_world_m(x, y)
        if self.goal_axis == "x":
            return ((x - self.pitch_x1) * self.px_to_m_x, (y - self.pitch_y1) * self.px_to_m_y)
        return ((y - self.pitch_y1) * self.px_to_m_y, (x - self.pitch_x1) * self.px_to_m_x)

    def _compute_xg(self, distance_m: float, angle_deg: float) -> float:
        """
        Simplified xG estimate via the standard two-feature approach (logistic
        regression on shot distance + angle only). The coefficients below are an
        illustrative approximation of the well-documented relationship - closer
        and more central shots score more often - NOT officially published/fitted
        values (no verifiably authoritative public coefficients were available).
        Treat this as a rough estimate, not broadcast-grade xG, which also factors
        defender/keeper positions, shot body part, and assist type.
        """
        logit = 1.2 - 0.10 * distance_m - 0.025 * angle_deg
        return 1.0 / (1.0 + math.exp(-logit))

    def _is_inside(self, pt: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _zone_for_team(self, team: str, bx: float, by: float) -> Optional[str]:
        """Which third of the pitch the ball falls in, from `team`'s own attacking
        perspective (their attacking third, not just "the high-coordinate side").
        Goes through _to_world_m (not a raw pixel fraction) so this reads
        correctly regardless of goal_axis - the along-pitch-length coordinate
        might come from pixel-x or pixel-y depending on camera orientation."""
        if team not in ("HOME", "AWAY"):
            return None
        world_len, _ = self._to_world_m(bx, by)
        x_frac = max(0.0, min(1.0, world_len / self.pitch_length_m))
        # x_frac measures "how far toward the high-length-coordinate end" -
        # flip it for whichever team attacks the LOW end, so it always reads
        # as "how far into this team's own attacking direction".
        team_attacks_left = (not self.home_attacks_right) if team == "HOME" else self.home_attacks_right
        if team_attacks_left:
            x_frac = 1.0 - x_frac
        if x_frac < 0.33:
            return "DEFENSIVE"
        if x_frac < 0.66:
            return "MIDDLE"
        return "ATTACKING"

    def process_frame(self, active_tracks: List[TrackedObject], frame_no: int) -> List[DetectedEvent]:
        timestamp_ms = int((frame_no / self.fps) * 1000)
        frame_events: List[DetectedEvent] = []

        ball = next((t for t in active_tracks if t.class_name == "ball"), None)
        players = [t for t in active_tracks if t.class_name == "player"]

        if not ball or len(players) == 0:
            return frame_events

        bx, by = ball.centroid

        # 1. Find nearest player to ball
        min_dist = float("inf")
        nearest_player: Optional[TrackedObject] = None

        for p in players:
            px, py = p.centroid
            dist = math.hypot(bx - px, by - py)
            if dist < min_dist:
                min_dist = dist
                nearest_player = p

        # Possession threshold (~60 pixels)
        possession_radius = 65.0
        current_possessor = nearest_player if min_dist < possession_radius else None

        if current_possessor:
            # Check for pass or turnover
            if self.last_possessor_id and self.last_possessor_id != current_possessor.track_id:
                if self.last_possessor_team == current_possessor.team:
                    # Completed pass
                    evt = DetectedEvent(
                        frame_no=frame_no,
                        timestamp_ms=timestamp_ms,
                        event_type="PASS",
                        player_track_id=self.last_possessor_id,
                        team=self.last_possessor_team,
                        data_json={
                            "from_player_track_id": self.last_possessor_id,
                            "to_player_track_id": current_possessor.track_id,
                            "confidence": 0.85,
                        },
                    )
                    frame_events.append(evt)
                else:
                    # Turnover / Possession change
                    evt = DetectedEvent(
                        frame_no=frame_no,
                        timestamp_ms=timestamp_ms,
                        event_type="POSSESSION_CHANGE",
                        player_track_id=current_possessor.track_id,
                        team=current_possessor.team,
                        data_json={
                            "from_team": self.last_possessor_team,
                            "to_team": current_possessor.team,
                        },
                    )
                    frame_events.append(evt)

            self.last_possessor_id = current_possessor.track_id
            self.last_possessor_team = current_possessor.team
            if current_possessor.team in self.possession_frames:
                self.possession_frames[current_possessor.team] += 1
            self.player_possession_frames[current_possessor.track_id] = (
                self.player_possession_frames.get(current_possessor.track_id, 0) + 1
            )

            zone = self._zone_for_team(current_possessor.team, bx, by)
            if zone is not None:
                self.zone_touches[current_possessor.team][zone] += 1

        # 2. Check for GOAL
        is_left_goal = self._is_inside((bx, by), self.left_goal)
        is_right_goal = self._is_inside((bx, by), self.right_goal)
        current_side = "LEFT" if is_left_goal else ("RIGHT" if is_right_goal else None)

        if current_side is not None and current_side == self._goal_zone_side:
            self._goal_zone_streak += 1
        else:
            self._goal_zone_side = current_side
            self._goal_zone_streak = 1 if current_side is not None else 0

        if (
            current_side is not None
            and self._goal_zone_streak >= self._goal_zone_required_streak
            and (frame_no - self.last_goal_frame) > (self.fps * 8)  # 8-second debounce
        ):
            # Whichever team's target goal is on this side scored - see
            # home_attacks_right for why this can't be hardcoded per side.
            home_scores_here = (current_side == "RIGHT") == self.home_attacks_right
            scoring_team = "HOME" if home_scores_here else "AWAY"

            goal_evt = DetectedEvent(
                frame_no=frame_no,
                timestamp_ms=timestamp_ms,
                event_type="GOAL",
                player_track_id=self.last_possessor_id,
                team=scoring_team,
                data_json={
                    "scoring_team": scoring_team,
                    "goal_side": current_side,
                    "scorer_track_id": self.last_possessor_id,
                    "confidence": 0.95,
                },
            )
            frame_events.append(goal_evt)
            self.last_goal_frame = frame_no

        # 3. Check for SHOT (ball suddenly moving fast, toward the attacking
        # team's target goal, from within realistic shooting range). Velocity is
        # measured over a bounded window (SHOT_MIN/MAX_WINDOW_SEC), not frame-to-
        # frame - see the constants above for why.
        min_gap = max(1, int(self.fps * self.SHOT_MIN_WINDOW_SEC))
        max_gap = int(self.fps * self.SHOT_MAX_WINDOW_SEC)

        if self._last_ball_pos is not None and self._last_ball_frame is not None:
            frame_diff = frame_no - self._last_ball_frame

            if frame_diff >= min_gap:
                if frame_diff <= max_gap:
                    prev_wx, prev_wy = self._to_world_m(*self._last_ball_pos)
                    curr_wx, curr_wy = self._to_world_m(bx, by)
                    dx_m, dy_m = curr_wx - prev_wx, curr_wy - prev_wy
                    vel_dist_m = math.hypot(dx_m, dy_m)
                    speed_kmh = (vel_dist_m / (frame_diff / self.fps)) * 3.6

                    shooter_team = self.last_possessor_team
                    target_goal = (
                        self.home_target_goal if shooter_team == "HOME"
                        else self.away_target_goal if shooter_team == "AWAY"
                        else None
                    )

                    if (
                        target_goal is not None
                        and self.SHOT_SPEED_KMH <= speed_kmh <= self.SHOT_MAX_SPEED_KMH
                        and (frame_no - self.last_shot_frame) > (self.fps * self.SHOT_COOLDOWN_SEC)
                    ):
                        goal_wx, goal_wy = self._to_world_m(*target_goal)
                        to_goal_x_m, to_goal_y_m = goal_wx - curr_wx, goal_wy - curr_wy
                        to_goal_dist_m = math.hypot(to_goal_x_m, to_goal_y_m)

                        cos_sim = (
                            (dx_m * to_goal_x_m + dy_m * to_goal_y_m) / (vel_dist_m * to_goal_dist_m)
                            if vel_dist_m > 0 and to_goal_dist_m > 0 else 0.0
                        )

                        if (
                            cos_sim >= self.SHOT_MIN_COS_SIM
                            and to_goal_dist_m <= self.SHOT_MAX_DIST_M
                        ):
                            # Angle off the goal's central axis - a standard simplified
                            # proxy for "how central/wide" the shot was (0 = straight on).
                            angle_deg = math.degrees(math.atan2(abs(to_goal_y_m), to_goal_dist_m))
                            xg = self._compute_xg(to_goal_dist_m, angle_deg)

                            shot_evt = DetectedEvent(
                                frame_no=frame_no,
                                timestamp_ms=timestamp_ms,
                                event_type="SHOT",
                                player_track_id=self.last_possessor_id,
                                team=shooter_team,
                                data_json={
                                    "distance_m": round(to_goal_dist_m, 2),
                                    "angle_deg": round(angle_deg, 2),
                                    "speed_kmh": round(speed_kmh, 1),
                                    "xg": round(xg, 3),
                                },
                            )
                            frame_events.append(shot_evt)
                            self.last_shot_frame = frame_no

                # Reference only advances once the minimum window has elapsed -
                # otherwise every single-frame step would keep resetting it and
                # velocity would never be measured over more than 1/fps seconds.
                self._last_ball_pos = (bx, by)
                self._last_ball_frame = frame_no
        else:
            self._last_ball_pos = (bx, by)
            self._last_ball_frame = frame_no

        self.events.extend(frame_events)
        return frame_events
