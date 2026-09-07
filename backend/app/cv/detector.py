import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DetectionResult:
    def __init__(self, class_name: str, bbox: List[float], confidence: float, class_id: int):
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
        }


class YOLOv8Detector:
    """
    YOLOv8 wrapper for football player and ball detection.

    Class IDs aren't assumed - they're read off the loaded model's own
    `names` mapping, since a COCO checkpoint (0=person, 32=sports ball) and
    our ball+player fine-tune (0=ball, 1=player) use different IDs for the
    same two concepts.
    """

    # Recognized model class names -> our internal label.
    NAME_MAP = {
        "person": "player",
        "player": "player",
        "sports ball": "ball",
        "ball": "ball",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        allow_fallback_on_empty: bool = False,
        img_size: int = 640,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Inference resolution. Counterintuitively, going above YOLO's native 640px on
        # real turf footage catches a few more far-field players per frame but makes
        # boxes jitter more frame-to-frame - net effect measured worse (more, not less,
        # track fragmentation) on real match video, so 640 is the tested default.
        self.img_size = img_size
        self.model = None
        # id -> our label ("player"/"ball"), populated from the model's own
        # names once loaded. self.ball_class_id is the model's own id for
        # "ball", used by detect_ball_tiled.
        self.class_map: Dict[int, str] = {}
        self.ball_class_id: Optional[int] = None
        # Synthetic simulation is only meaningful for the drawn-shape demo video (no real
        # COCO objects to detect there). On real footage, a genuinely empty frame (blank
        # intro card, no one in view, ball out of shot) must stay empty - not be replaced
        # with fabricated player positions.
        self.allow_fallback_on_empty = allow_fallback_on_empty

        # Auto-detect best hardware acceleration
        if device:
            self.device = device
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model from {self.model_path} (device: {self.device})...")
            self.model = YOLO(self.model_path)
            # Transfer model to device
            if self.device != "cpu":
                self.model.to(self.device)

            self.class_map = {
                cls_id: self.NAME_MAP[name]
                for cls_id, name in self.model.names.items()
                if name in self.NAME_MAP
            }
            self.ball_class_id = next(
                (cls_id for cls_id, label in self.class_map.items() if label == "ball"),
                None,
            )
            logger.info(f"YOLOv8 model loaded successfully. class_map={self.class_map}")
        except Exception as e:
            logger.warning(f"Could not load ultralytics YOLO model ({e}). Will use fallback/synthetic detector.")
            self.model = None

    def detect_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on a single BGR OpenCV frame.
        Returns a list of DetectionResult objects.
        """
        if self.model is None:
            return self._fallback_detect(frame)

        try:
            # Run inference with GPU acceleration
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=list(self.class_map.keys()),
                device=self.device,
                imgsz=self.img_size,
                verbose=False,
            )

            detections: List[DetectionResult] = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    class_name = self.class_map.get(cls_id, "unknown")

                    detections.append(
                        DetectionResult(
                            class_name=class_name,
                            bbox=xyxy,
                            confidence=conf,
                            class_id=cls_id,
                        )
                    )
            # On the synthetic demo canvas, the real model finds no COCO objects (they're
            # drawn shapes) - supplement with simulation there only, never on real footage.
            if len(detections) == 0 and self.allow_fallback_on_empty:
                return self._fallback_detect(frame)
            return self._dedup_overlapping_player_detections(detections)
        except Exception as e:
            logger.error(f"Inference error with YOLOv8: {e}")
            if self.allow_fallback_on_empty:
                return self._fallback_detect(frame)
            return []

    def detect_ball_tiled(
        self,
        frame: np.ndarray,
        grid: Tuple[int, int] = (2, 2),
        overlap: float = 0.15,
        conf_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Ball-only detection over overlapping tiles of the frame, effectively boosting
        the model's resolution for the single hardest class to see at broadcast camera
        distance (a football is a handful of low-contrast, often motion-blurred pixels
        in a 1080p+ wide shot - well outside YOLO's native 640px input). Players don't
        need this: a single full-frame pass already finds them reliably.

        Meaningfully more expensive than a single pass (cols*rows extra inference calls
        per frame) - only worth it when ball detection actually matters for the match
        (goal/pass/possession events depend on it).
        """
        if self.model is None or self.ball_class_id is None:
            return []

        # Ball detection lives in a much lower confidence regime than player detection on
        # this kind of footage - measured hit rate goes from ~0% at 0.15+ to a real (if
        # still sparse) ~13% at 0.08 on actual match video. Tuned empirically, not a
        # fraction of the (unrelated) player confidence threshold.
        conf = conf_threshold if conf_threshold is not None else 0.08
        h, w = frame.shape[:2]
        cols, rows = grid
        tile_w, tile_h = w / cols, h / rows
        pad_w, pad_h = tile_w * overlap, tile_h * overlap

        all_dets: List[DetectionResult] = []
        for r in range(rows):
            for c in range(cols):
                x1 = max(0, int(c * tile_w - pad_w))
                y1 = max(0, int(r * tile_h - pad_h))
                x2 = min(w, int((c + 1) * tile_w + pad_w))
                y2 = min(h, int((r + 1) * tile_h + pad_h))
                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                try:
                    results = self.model(
                        tile,
                        conf=conf,
                        iou=self.iou_threshold,
                        classes=[self.ball_class_id],
                        device=self.device,
                        imgsz=self.img_size,
                        verbose=False,
                    )
                except Exception as e:
                    logger.error(f"Tiled ball inference error: {e}")
                    continue

                for res in results:
                    for box in res.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        all_dets.append(
                            DetectionResult(
                                class_name="ball",
                                bbox=[bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                                confidence=float(box.conf[0].item()),
                                class_id=self.ball_class_id,
                            )
                        )

        return self._dedup_ball_detections(all_dets)

    def _dedup_ball_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Overlapping tiles can both catch a ball near their shared border - keep the highest-confidence box."""
        if len(detections) <= 1:
            return detections
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[DetectionResult] = []
        for d in detections:
            if not any(self._iou(d.bbox, k.bbox) > 0.3 for k in kept):
                kept.append(d)
        return kept

    # Confirmed on real production footage: an awkward pose (an arm raised mid-
    # header/reach) occasionally makes the model output a second, smaller, lower-
    # confidence box nested almost entirely inside the real one - e.g. just the
    # raised arm/shoulder read as its own "player". These two boxes are such
    # different sizes that their IoU is small (measured ~0.13 on the real case),
    # so the model's own NMS (which gates on IoU) doesn't merge them - but 96% of
    # the smaller box's own area sits inside the larger one, which is what actually
    # identifies "these are the same detection", not IoU. Left undeduped, the
    # tracker matched both to separate tracks with conflicting team labels.
    PLAYER_CONTAINMENT_DEDUP_THRESHOLD = 0.85

    def _dedup_overlapping_player_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Drops a lower-confidence 'player' detection that's almost entirely contained within a
        higher-confidence one (see PLAYER_CONTAINMENT_DEDUP_THRESHOLD). Ball detections pass through
        untouched - there's only ever one ball, and its own tracking already handles duplicates."""
        players = [d for d in detections if d.class_name == "player"]
        others = [d for d in detections if d.class_name != "player"]
        if len(players) <= 1:
            return detections

        players = sorted(players, key=lambda d: d.confidence, reverse=True)
        kept: List[DetectionResult] = []
        for d in players:
            if not any(
                self._containment_ratio(d.bbox, k.bbox) > self.PLAYER_CONTAINMENT_DEDUP_THRESHOLD
                for k in kept
            ):
                kept.append(d)
        return others + kept

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _containment_ratio(a: List[float], b: List[float]) -> float:
        """Fraction of the SMALLER box's own area that overlaps the other - unlike IoU,
        correctly flags a small box nested inside a much larger one as "the same thing"."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller = min(area_a, area_b)
        return inter / smaller if smaller > 0 else 0.0

    def _fallback_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Synthetic fallback detector for testing without heavy weights or on drawn synthetic canvas.
        Generates continuous moving player and ball detections with passes and a goal.
        """
        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        detections = []

        if not hasattr(self, "_fallback_frame_count"):
            self._fallback_frame_count = 0
        self._fallback_frame_count += 1
        # Normalize over 75 processed frames
        t = min(1.0, (self._fallback_frame_count % 75) / 74.0)

        # 10 simulated players (5 HOME on left/center, 5 AWAY on right/center)
        for i in range(10):
            team_offset = 0.20 if i < 5 else 0.60
            base_cx = (team_offset + (i % 5) * 0.08) * w
            base_cy = (0.25 + (i % 5) * 0.12) * h
            # Dynamic movement
            cx = base_cx + 25 * np.sin(2 * np.pi * t * 2 + i)
            cy = base_cy + 15 * np.cos(2 * np.pi * t * 2 + i)
            bw, bh = 40, 80
            detections.append(
                DetectionResult(
                    class_name="player",
                    bbox=[max(0, cx - bw / 2), max(0, cy - bh / 2), min(w, cx + bw / 2), min(h, cy + bh / 2)],
                    confidence=0.88,
                    class_id=0,
                )
            )

        # Ball trajectory: starts near Player #1, passes to Player #3, then strikes into right goal
        if t < 0.35:
            # Near Player #1
            p1_x = (0.20 + 0 * 0.08) * w + 25 * np.sin(2 * np.pi * t * 2)
            p1_y = (0.25 + 0 * 0.12) * h
            bx = p1_x + 10
            by = p1_y + 10
        elif t < 0.70:
            # Pass to Player #3
            sub_t = (t - 0.35) / 0.35
            p3_x = (0.20 + 2 * 0.08) * w
            p3_y = (0.25 + 2 * 0.12) * h
            bx = ((0.20 * w) + (p3_x - 0.20 * w) * sub_t)
            by = ((0.25 * h) + (p3_y - 0.25 * h) * sub_t)
        else:
            # Shot into Right Goal
            sub_t = (t - 0.70) / 0.30
            start_x = (0.20 + 2 * 0.08) * w
            start_y = (0.25 + 2 * 0.12) * h
            goal_x = w * 0.95
            goal_y = h * 0.50
            bx = start_x + (goal_x - start_x) * sub_t
            by = start_y + (goal_y - start_y) * sub_t

        detections.append(
            DetectionResult(
                class_name="ball",
                bbox=[bx - 10, by - 10, bx + 10, by + 10],
                confidence=0.92,
                class_id=32,
            )
        )
        return detections
