from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class EventCreate(BaseModel):
    match_id: str
    frame_no: int
    timestamp_ms: int = 0
    event_type: str = Field(..., description="GOAL, SHOT, POSSESSION_CHANGE, FOUL")
    player_id: Optional[str] = None
    team: Optional[str] = None
    data_json: Dict[str, Any] = Field(default_factory=dict)


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    match_id: str
    frame_no: int
    timestamp_ms: int
    event_type: str
    player_id: Optional[str] = None
    team: Optional[str] = None
    data_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

