import math
from typing import List, Dict, Any, Tuple
import numpy as np
from app.cv.tracker import TrackedObject
from app.cv.events import DetectedEvent


class StatsCalculator:
    """
    Computes spatial and physiological match analytics:
    - 2D Heatmaps
    - Distance covered (meters)
    - Top speed (km/h)
    - Pass count & possession percentage
    """

    def __init__(
        self,
        pitch_length_m: float = 40.0,
        pitch_width_m: float = 20.0,
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: float = 30.0,
    ):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # Scaling factors (pixels to meters)
        self.px_to_m_x = pitch_length_m / frame_width
        self.px_to_m_y = pitch_width_m / frame_height

    def compute_player_stats(
        self,
        player_track: TrackedObject,
        events: List[DetectedEvent],
        total_frames: int,
    ) -> Dict[str, Any]:
        """
        Computes detailed stats for a single tracked player.
        """
        history = player_track.history  # List of (frame_no, (cx, cy))
        if len(history) < 2:
            return {
                "distance_m": 0.0,
                "top_speed_kmh": 0.0,
                "passes": 0,
                "shots": 0,
                "possession_pct": 0.0,
                "heatmap_json": [],
            }

        # 1. Total Distance (in meters)
        total_distance_m = 0.0
        speeds_kmh: List[float] = []

        # Calculate distances and speeds using rolling window
        for i in range(1, len(history)):
            f_prev, (x1, y1) = history[i - 1]
            f_curr, (x2, y2) = history[i]

            dx_m = (x2 - x1) * self.px_to_m_x
            dy_m = (y2 - y1) * self.px_to_m_y
            dist_m = math.hypot(dx_m, dy_m)
            total_distance_m += dist_m

            frame_diff = f_curr - f_prev
            if frame_diff > 0:
                time_sec = frame_diff / self.fps
                speed_mps = dist_m / time_sec
                speed_kmh = speed_mps * 3.6
                # Filter noise/teleportation artifacts (> 35 km/h)
                if speed_kmh < 35.0:
                    speeds_kmh.append(speed_kmh)

        top_speed_kmh = round(float(np.max(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0, 2)
        total_distance_m = round(float(total_distance_m), 2)

        # 2. 2D Heatmap (20 rows x 30 columns grid)
        heatmap_grid = self._generate_heatmap_grid([pt for _, pt in history], rows=20, cols=30)

        # 3. Events attributed to player
        player_passes = sum(
            1 for e in events if e.event_type == "PASS" and e.player_track_id == player_track.track_id
        )
        player_shots = sum(
            1 for e in events if e.event_type in ["SHOT", "GOAL"] and e.player_track_id == player_track.track_id
        )

        # 4. Estimated possession percentage
        # Fraction of events or time involved
        possession_events = sum(
            1 for e in events if e.player_track_id == player_track.track_id
        )
        total_events = max(1, len(events))
        possession_pct = round((possession_events / total_events) * 100, 1)

        return {
            "distance_m": total_distance_m,
            "top_speed_kmh": top_speed_kmh,
            "passes": player_passes,
            "shots": player_shots,
            "possession_pct": possession_pct,
            "heatmap_json": heatmap_grid,
        }

    def _generate_heatmap_grid(
        self,
        positions: List[Tuple[float, float]],
        rows: int = 20,
        cols: int = 30
    ) -> List[List[float]]:
        """
        Accumulates player position coordinates into a normalized 2D grid matrix [0.0 - 1.0].
        """
        grid = np.zeros((rows, cols), dtype=float)

        for x, y in positions:
            c = int((x / self.frame_width) * cols)
            r = int((y / self.frame_height) * rows)
            c = max(0, min(cols - 1, c))
            r = max(0, min(rows - 1, r))
            grid[r, c] += 1.0

        max_val = np.max(grid)
        if max_val > 0:
            grid = grid / max_val

        return [[round(float(val), 3) for val in row] for row in grid]
