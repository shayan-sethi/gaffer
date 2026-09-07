import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.cv.detector import DetectionResult
from app.cv.tracker import TeamAssignmentMixin, TrackedObject

# gaffer_botsort.yaml overrides only track_buffer (30 -> 450) from ultralytics'
# own botsort.yaml default - see that file's header comment for why.
DEFAULT_TRACKER_YAML = os.path.join(os.path.dirname(__file__), "gaffer_botsort.yaml")


class BotSortPlayerTracker(TeamAssignmentMixin):
    """
    Player tracking via ultralytics' built-in BoT-SORT tracker, as an alternative
    to CentroidTracker's hand-rolled centroid+Hungarian matching. BoT-SORT brings
    two things that were hard to get right by hand in this project: a proper
    occlusion-aware Kalman motion model with IoU-based re-association (tuned
    across a huge range of real multi-object-tracking benchmarks, not just one
    match), and built-in camera-motion compensation (`gmc_method: sparseOptFlow`
    in botsort.yaml) - the exact feature a hand-rolled attempt earlier in this
    project regressed real results and had to be reverted.

    Uses gaffer_botsort.yaml (see DEFAULT_TRACKER_YAML), not ultralytics' own
    botsort.yaml, by default - confirmed on real footage that the stock 30-call
    track_buffer gives up on a lost player (camera pan, brief occlusion, a
    player leaving/re-entering frame) far more readily than this project's own
    CentroidTracker: 204 player track IDs over a 10-minute clip with ~12-14 real
    players, vs 17-18 with CentroidTracker on the same window. A much longer
    buffer gives BoT-SORT's own motion model more chances to re-associate a
    reappearing player via IoU/motion alone, rather than minting a new identity.

    Ball tracking is deliberately NOT delegated to BoT-SORT: a general-purpose
    multi-object tracker has no notion of "there's only ever one real ball", so
    it would manage multiple candidate ball tracks exactly like it does players.
    update() returns the frame's raw ball DetectionResults (BoT-SORT's own ID for
    them discarded, not reused) for the caller to feed into CentroidTracker's
    existing single-ball continuity logic instead (see CentroidTracker._update_ball).

    Loads its OWN model instance from `model_path`, rather than reusing an
    already-loaded one - confirmed on real footage that a single ultralytics
    YOLO object's internal predictor/tracker state gets silently corrupted the
    moment a *plain* (non-tracking) inference call is made on that same object
    (e.g. YOLOv8Detector.detect_ball_tiled's tiled ball-only fallback, which
    fires on nearly every frame since ball detections are sparse): every
    .track() call after that returned zero boxes. Two independent model
    objects - loading the same ~6MB weights file twice - sidesteps this
    entirely, and is cheap enough not to matter.
    """

    def __init__(
        self,
        model_path: str,
        class_map: Dict[int, str],
        tracker_yaml: str = DEFAULT_TRACKER_YAML,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        if device != "cpu":
            self.model.to(device)
        self.class_map = class_map
        self.tracker_yaml = tracker_yaml
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.tracks: Dict[int, TrackedObject] = {}
        self._team_reference_centroids: Optional[np.ndarray] = None
        self._team_assign_counter = 0

    def update(
        self,
        frame: np.ndarray,
        frame_no: int,
        pitch_mask=None,
        within_pitch=None,
    ) -> Tuple[List[TrackedObject], List[DetectionResult], List[DetectionResult]]:
        """
        Runs one YOLO forward pass via the model's own .track() - detection and
        BoT-SORT association happen together internally; the public API doesn't
        allow feeding it pre-computed detections separately. Returns (active
        player tracks, kept player DetectionResults, raw ball DetectionResults).

        The player DetectionResults mirror exactly what YOLOv8Detector.detect_frame
        would have returned for the players that passed pitch_mask - so callers
        can build Detection ORM rows via the same bbox-matching-against-active-
        tracks pattern already used for the CentroidTracker path, unchanged.

        `pitch_mask`, if given, drops player boxes outside the field of play
        (sideline spectators/bystanders) exactly like track.py already does for
        the CentroidTracker path - applied here (after tracking) rather than
        before, since BoT-SORT's own detect step can't be intercepted separately.

        `within_pitch`, if given, is a second `(cx, cy) -> bool` geometric gate
        (see PitchCalibration.contains_point_px) - a calibrated camera's precise
        real-world pitch rectangle, independent of pitch_mask's color-based
        segmentation. Kept as a plain callable rather than importing
        PitchCalibration here, so this module doesn't need to know how that
        check is implemented or whether the camera is even calibrated at all.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_yaml,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes

        active: List[TrackedObject] = []
        player_dets: List[DetectionResult] = []
        ball_dets: List[DetectionResult] = []
        if boxes is None or boxes.id is None:
            # No confirmed tracks yet this frame (e.g. the very first frame, or
            # every detection fell below BoT-SORT's own new_track_thresh).
            self._assign_teams()
            return active, player_dets, ball_dets

        ids = boxes.id.int().tolist()
        clss = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        xyxys = boxes.xyxy.tolist()

        for track_id, cls_id, conf, xyxy in zip(ids, clss, confs, xyxys):
            class_name = self.class_map.get(cls_id, "unknown")

            if class_name == "ball":
                ball_dets.append(DetectionResult("ball", xyxy, conf, cls_id))
                continue
            if class_name != "player":
                continue

            # Bottom-center, matching CentroidTracker._get_centroid's convention
            # for players (ground footprint, not box center).
            cx, cy = (xyxy[0] + xyxy[2]) / 2.0, xyxy[3]
            if pitch_mask is not None and not pitch_mask.contains_point(cx, cy):
                continue
            if within_pitch is not None and not within_pitch(cx, cy):
                continue

            color = self._sample_color(frame, xyxy)
            if track_id in self.tracks:
                self.tracks[track_id].update(xyxy, (cx, cy), frame_no, color_sample=color)
            else:
                t = TrackedObject(
                    track_id=track_id, class_name="player", bbox=xyxy, centroid=(cx, cy), frame_no=frame_no,
                )
                if color is not None:
                    t.color_samples.append(color)
                self.tracks[track_id] = t
            active.append(self.tracks[track_id])
            player_dets.append(DetectionResult("player", xyxy, conf, cls_id))

        self._assign_teams()
        return active, player_dets, ball_dets

    def prune_unconfirmed_tracks(self, min_hits: int = 2) -> None:
        """
        Drops tracks that were only ever matched once (a single-frame detector
        false positive that BoT-SORT still confidently minted a track ID for,
        given its default new_track_thresh=0.25) - confirmed on real footage to
        be a meaningful share (~15%) of its inflated track count, distinct from
        the larger fragmentation problem `track_buffer` addresses. Call once,
        after tracking is complete for the whole video - a track only reaches
        its final hit count then, so pruning any earlier risks dropping one that
        would have gone on to be matched again.
        """
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.hits >= min_hits}
