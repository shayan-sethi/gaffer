from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class StartMatchRequest(BaseModel):
    turf_id: Optional[str] = Field(None, description="Turf identifier")
    title: Optional[str] = Field("5-a-side Match", description="Match title")
    expected_duration_sec: Optional[int] = Field(3600, description="Expected match duration in seconds")
    meta_data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UploadCompleteRequest(BaseModel):
    match_id: str = Field(..., description="ID of the finished match")
    s3_url: str = Field(..., description="Uploaded video S3 or MinIO URL")
    duration_sec: Optional[int] = Field(0, description="Actual recorded duration in seconds")
    file_size_bytes: Optional[int] = Field(None, description="Uploaded file size")


class TriggerProcessRequest(BaseModel):
    video_s3_url: Optional[str] = Field(None, description="Custom video S3 URL to process")


class PlayerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    match_id: str
    track_id: int
    jersey_number: Optional[int] = None
    player_name: str
    team: str


class MatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    turf_id: Optional[str] = None
    camera_id: Optional[str] = None
    title: str
    date: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_sec: int
    video_s3_url: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    meta_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class MatchDetailResponse(MatchResponse):
    players: List[PlayerResponse] = []

