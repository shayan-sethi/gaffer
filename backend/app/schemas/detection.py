from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class DetectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    match_id: str
    frame_no: int
    timestamp_ms: int
    track_id: Optional[int] = None
    class_name: str
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float
    created_at: datetime

