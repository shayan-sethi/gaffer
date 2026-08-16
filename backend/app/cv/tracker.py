import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from app.cv.detector import DetectionResult


class TrackedObject:
    def __init__(self, track_id: int, class_name: str, bbox: List[float], centroid: Tuple[float, float], frame_no: int):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.centroid = centroid  # (cx, cy) - usually bottom-center for players
        self.history: List[Tuple[int, Tuple[float, float]]] = [(frame_no, centroid)]  # (frame_no, centroid)
        self.consecutive_lost = 0
        self.team: str = "UNASSIGNED"  # HOME, AWAY, REFEREE
        self.total_distance_px: float = 0.0

    def update(self, bbox: List[float], centroid: Tuple[float, float], frame_no: int):
        # Calculate distance moved in pixels
        prev_centroid = self.centroid
        dx = centroid[0] - prev_centroid[0]
        dy = centroid[1] - prev_centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)
        self.total_distance_px += dist

        self.bbox = bbox
        self.centroid = centroid
        self.history.append((frame_no, centroid))
        self.consecutive_lost = 0

    def mark_lost(self):
        self.consecutive_lost += 1


class CentroidTracker:
    """
    Centroid & IoU-assisted multi-object tracker for football players & ball.
    Maintains persistent track IDs across frames.
    """

    def __init__(self, max_lost_frames: int = 30, max_distance_px: float = 80.0):
        self.max_lost_frames = max_lost_frames
        self.max_distance_px = max_distance_px
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def _get_centroid(self, bbox: List[float], class_name: str) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        if class_name == "player":
            # For players, bottom-center represents ground footprint on pitch
            return ((x1 + x2) / 2.0, y2)
        else:
            # For ball, geometric center
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: List[DetectionResult], frame_no: int) -> List[TrackedObject]:
        """
        Update tracker with detections from the current frame.
        Returns list of active TrackedObjects for this frame.
        """
        # Separate player detections from ball
        player_dets = [d for d in detections if d.class_name == "player"]
        ball_dets = [d for d in detections if d.class_name == "ball"]

        active_tracks: List[TrackedObject] = []

        # 1. Match player detections to existing player tracks
        player_tracks = {t_id: t for t_id, t in self.tracks.items() if t.class_name == "player"}
        matched_tracks = self._match_detections(player_tracks, player_dets, frame_no)
        active_tracks.extend(matched_tracks)

        # 2. Match or track ball
        ball_tracks = {t_id: t for t_id, t in self.tracks.items() if t.class_name == "ball"}
        matched_ball_tracks = self._match_detections(ball_tracks, ball_dets, frame_no, max_dist=150.0)
        active_tracks.extend(matched_ball_tracks)

        # 3. Clean up stale tracks
        expired_ids = [
            t_id for t_id, t in self.tracks.items()
            if t.consecutive_lost > self.max_lost_frames
        ]
        for t_id in expired_ids:
            del self.tracks[t_id]

        # 4. Assign teams based on average pitch coordinates
        self._assign_teams()

        return active_tracks

    def _match_detections(
        self,
        existing_tracks: Dict[int, TrackedObject],
        detections: List[DetectionResult],
        frame_no: int,
        max_dist: Optional[float] = None
    ) -> List[TrackedObject]:
        max_distance = max_dist if max_dist is not None else self.max_distance_px
        updated_tracks: List[TrackedObject] = []

        if len(existing_tracks) == 0:
            for det in detections:
                centroid = self._get_centroid(det.bbox, det.class_name)
                track = TrackedObject(
                    track_id=self.next_track_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    centroid=centroid,
                    frame_no=frame_no,
                )
                self.tracks[self.next_track_id] = track
                updated_tracks.append(track)
                self.next_track_id += 1
            return updated_tracks

        track_ids = list(existing_tracks.keys())
        track_centroids = [existing_tracks[t_id].centroid for t_id in track_ids]

        det_centroids = [self._get_centroid(d.bbox, d.class_name) for d in detections]

        if len(det_centroids) == 0:
            for t_id in track_ids:
                existing_tracks[t_id].mark_lost()
            return updated_tracks

        # Compute cost matrix (Euclidean distance)
        cost_matrix = np.zeros((len(track_ids), len(det_centroids)), dtype=float)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                cost_matrix[i, j] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

        # Greedy / Row-column minimum assignment
        rows = cost_matrix.min(axis=1).argsort()
        cols = cost_matrix.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue

            if cost_matrix[r, c] > max_distance:
                continue

            t_id = track_ids[r]
            det = detections[c]
            centroid = det_centroids[c]

            existing_tracks[t_id].update(det.bbox, centroid, frame_no)
            updated_tracks.append(existing_tracks[t_id])

            assigned_rows.add(r)
            assigned_cols.add(c)

        # Mark unmatched existing tracks as lost
        for i, t_id in enumerate(track_ids):
            if i not in assigned_rows:
                existing_tracks[t_id].mark_lost()

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in assigned_cols:
                centroid = det_centroids[j]
                track = TrackedObject(
                    track_id=self.next_track_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    centroid=centroid,
                    frame_no=frame_no,
                )
                self.tracks[self.next_track_id] = track
                updated_tracks.append(track)
                self.next_track_id += 1

        return updated_tracks

    def _assign_teams(self):
        """
        Assigns HOME / AWAY teams to player tracks based on spatial clustering.
        """
        players = [t for t in self.tracks.values() if t.class_name == "player"]
        if len(players) < 2:
            return

        x_coords = [p.centroid[0] for p in players]
        median_x = np.median(x_coords)

        for p in players:
            if p.team == "UNASSIGNED":
                if p.centroid[0] < median_x:
                    p.team = "HOME"
                else:
                    p.team = "AWAY"
