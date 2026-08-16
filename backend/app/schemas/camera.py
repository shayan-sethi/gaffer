from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class CameraRegisterRequest(BaseModel):
    serial_number: str = Field(..., description="Unique hardware serial number")
    turf_name: str = Field(..., description="Turf or facility display name")
    turf_id: Optional[str] = Field(None, description="Optional Turf ID")
    ip_address: Optional[str] = Field(None, description="Local IP address of camera")
    firmware_version: Optional[str] = "1.0.0"


class CameraHeartbeatRequest(BaseModel):
    status: Optional[str] = "ONLINE"
    ip_address: Optional[str] = None
    firmware_version: Optional[str] = None


class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    serial_number: str
    turf_name: str
    turf_id: Optional[str] = None
    api_key: str
    status: str
    ip_address: Optional[str] = None
    firmware_version: str
    last_heartbeat: Optional[datetime] = None
    created_at: datetime

