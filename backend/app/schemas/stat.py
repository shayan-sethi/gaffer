from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ZoneTouches(BaseModel):
    DEFENSIVE: int = 0
    MIDDLE: int = 0
    ATTACKING: int = 0


class PlayerStatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    match_id: str
    player_id: str
    jersey_number: Optional[int] = None
    player_name: Optional[str] = None
    team: Optional[str] = None
    distance_m: float
    top_speed_kmh: float
    avg_speed_kmh: float = 0.0
    sprints: int = 0
    passes: int
    shots: int
    possession_pct: float
    heatmap_json: List[List[float]] = Field(default_factory=list, description="2D matrix or coordinate density points")
    updated_at: datetime



class MatchStatsOverviewResponse(BaseModel):
    match_id: str
    duration_sec: int
    home_score: int
    away_score: int
    home_possession_pct: float
    away_possession_pct: float
    total_distance_m: float
    total_shots: int
    total_passes: int
    zone_touches: Optional[Dict[str, ZoneTouches]] = None
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_corners: int = 0
    away_corners: int = 0
    player_stats: List[PlayerStatResponse] = []
