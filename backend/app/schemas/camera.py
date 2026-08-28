from datetime import datetime
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator


class CameraCalibrationRequest(BaseModel):
    # Exactly 4 pixel points, in TL, TR, BR, BL order around the pitch, marked
    # against a reference frame from this camera (e.g. a frame grab from its
    # most recent match) - see PitchCalibration for why the order matters.
    points: List[Tuple[float, float]] = Field(..., min_length=4, max_length=4)

    @field_validator("points")
    @classmethod
    def points_not_collinear(cls, points):
        xs = {p[0] for p in points}
        ys = {p[1] for p in points}
        if len(xs) == 1 or len(ys) == 1:
            raise ValueError("Calibration points cannot be collinear (all same x or all same y).")
        return points


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
    calibration_points: Optional[List[List[float]]] = None
    last_heartbeat: Optional[datetime] = None
    created_at: datetime

