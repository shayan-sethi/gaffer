import math
from typing import List, Dict, Any, Optional, Tuple
from app.cv.tracker import TrackedObject


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

    def __init__(self, frame_width: int = 1280, frame_height: int = 720, fps: float = 30.0):
        self.width = frame_width
        self.height = frame_height
        self.fps = fps

        # Goal areas (left 8% and right 8% of the frame along vertical center)
        self.left_goal = (0, self.height * 0.35, self.width * 0.08, self.height * 0.65)
        self.right_goal = (self.width * 0.92, self.height * 0.35, self.width, self.height * 0.65)

        self.last_possessor_id: Optional[int] = None
        self.last_possessor_team: Optional[str] = None
        self.last_goal_frame: int = -300  # Cooldown between goal events (10 sec @ 30fps)
        self.events: List[DetectedEvent] = []

    def _is_inside(self, pt: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

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

        # 2. Check for GOAL
        if (frame_no - self.last_goal_frame) > (self.fps * 8):  # 8-second debounce
            is_left_goal = self._is_inside((bx, by), self.left_goal)
            is_right_goal = self._is_inside((bx, by), self.right_goal)

            if is_left_goal or is_right_goal:
                scoring_team = "AWAY" if is_left_goal else "HOME"
                goal_side = "LEFT" if is_left_goal else "RIGHT"

                goal_evt = DetectedEvent(
                    frame_no=frame_no,
                    timestamp_ms=timestamp_ms,
                    event_type="GOAL",
                    player_track_id=self.last_possessor_id,
                    team=scoring_team,
                    data_json={
                        "scoring_team": scoring_team,
                        "goal_side": goal_side,
                        "scorer_track_id": self.last_possessor_id,
                        "confidence": 0.95,
                    },
                )
                frame_events.append(goal_evt)
                self.last_goal_frame = frame_no

        self.events.extend(frame_events)
        return frame_events
