from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


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
    player_stats: List[PlayerStatResponse] = []
