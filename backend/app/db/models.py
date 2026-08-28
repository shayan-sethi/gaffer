import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Enum,
    Text,
)
from sqlalchemy.orm import relationship
import enum

from app.db.database import Base


class MatchStatus(str, enum.Enum):
    PENDING = "PENDING"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CameraStatus(str, enum.Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    RECORDING = "RECORDING"
    ERROR = "ERROR"


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    serial_number = Column(String(100), unique=True, index=True, nullable=False)
    turf_name = Column(String(200), nullable=False)
    turf_id = Column(String(100), index=True, nullable=True)
    api_key = Column(String(200), unique=True, index=True, nullable=False)
    status = Column(Enum(CameraStatus), default=CameraStatus.OFFLINE)
    ip_address = Column(String(50), nullable=True)
    firmware_version = Column(String(50), default="1.0.0")
    last_heartbeat = Column(DateTime, nullable=True)
    # 4 pixel points [[x,y], ...] marking the pitch corners in a reference frame,
    # in TL, TR, BR, BL order (matching the pitch_x=0/pitch_y=0 top-left
    # convention used throughout the CV pipeline) - set once per fixed camera
    # install so distance/speed can be computed via a real perspective
    # correction instead of a flat pixel-to-meter ratio. Null until an operator
    # calibrates the camera.
    calibration_points = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    matches = relationship("Match", back_populates="camera")


class Match(Base):
    __tablename__ = "matches"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    turf_id = Column(String(100), index=True, nullable=True)
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=True)
    title = Column(String(200), default="5-a-side Match")
    date = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    duration_sec = Column(Integer, default=0)
    video_s3_url = Column(String(500), nullable=True)
    status = Column(Enum(MatchStatus), default=MatchStatus.PENDING, index=True)
    error_message = Column(Text, nullable=True)
    meta_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    camera = relationship("Camera", back_populates="matches")
    players = relationship("Player", back_populates="match", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="match", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="match", cascade="all, delete-orphan")
    player_stats = relationship("PlayerStat", back_populates="match", cascade="all, delete-orphan")


class Player(Base):
    __tablename__ = "players"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String(36), ForeignKey("matches.id"), index=True, nullable=False)
    track_id = Column(Integer, nullable=False, index=True)
    jersey_number = Column(Integer, nullable=True)
    player_name = Column(String(100), default="Player")
    team = Column(String(50), default="HOME")  # HOME, AWAY, REFEREE, UNASSIGNED
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="players")
    stats = relationship("PlayerStat", back_populates="player", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String(36), ForeignKey("matches.id"), index=True, nullable=False)
    frame_no = Column(Integer, nullable=False, index=True)
    timestamp_ms = Column(Integer, default=0)
    track_id = Column(Integer, nullable=True, index=True)
    class_name = Column(String(50), nullable=False)  # player, ball, referee
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="detections")


class Event(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String(36), ForeignKey("matches.id"), index=True, nullable=False)
    frame_no = Column(Integer, nullable=False, index=True)
    timestamp_ms = Column(Integer, default=0)
    event_type = Column(String(50), nullable=False, index=True)  # GOAL, SHOT, POSSESSION_CHANGE, FOUL
    player_id = Column(String(36), ForeignKey("players.id"), nullable=True)
    team = Column(String(50), nullable=True)
    data_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="events")


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    match_id = Column(String(36), ForeignKey("matches.id"), index=True, nullable=False)
    player_id = Column(String(36), ForeignKey("players.id"), nullable=False)
    distance_m = Column(Float, default=0.0)
    top_speed_kmh = Column(Float, default=0.0)
    avg_speed_kmh = Column(Float, default=0.0)
    sprints = Column(Integer, default=0)
    passes = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    possession_pct = Column(Float, default=0.0)
    heatmap_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    match = relationship("Match", back_populates="player_stats")
    player = relationship("Player", back_populates="stats")
