import secrets
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Camera, CameraStatus, Match, MatchStatus
from app.schemas.camera import CameraRegisterRequest, CameraHeartbeatRequest, CameraResponse, CameraCalibrationRequest
from app.schemas.match import StartMatchRequest, UploadCompleteRequest, MatchResponse
from app.tasks.pipeline import trigger_match_pipeline
from app.cv.calibration import PitchCalibration

router = APIRouter(prefix="/cameras", tags=["Cameras"])


def verify_camera_auth(x_camera_key: Optional[str] = Header(None), db: Session = Depends(get_db)) -> Optional[Camera]:
    """Helper to authenticate camera requests if API key is provided."""
    if not x_camera_key:
        return None
    cam = db.query(Camera).filter(Camera.api_key == x_camera_key).first()
    return cam


@router.post("/register", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def register_camera(payload: CameraRegisterRequest, db: Session = Depends(get_db)):
    """
    Turf operator or factory pairs/registers a Gaffer camera with the cloud.
    Returns a secure camera API key for subsequent daemon requests.
    """
    existing = db.query(Camera).filter(Camera.serial_number == payload.serial_number).first()
    if existing:
        # Update existing camera registration
        existing.turf_name = payload.turf_name
        if payload.turf_id:
            existing.turf_id = payload.turf_id
        if payload.ip_address:
            existing.ip_address = payload.ip_address
        existing.last_heartbeat = datetime.utcnow()
        existing.status = CameraStatus.ONLINE
        db.commit()
        db.refresh(existing)
        return existing

    api_key = f"gaffer_cam_{secrets.token_urlsafe(32)}"
    camera = Camera(
        serial_number=payload.serial_number,
        turf_name=payload.turf_name,
        turf_id=payload.turf_id,
        api_key=api_key,
        status=CameraStatus.ONLINE,
        ip_address=payload.ip_address,
        firmware_version=payload.firmware_version or "1.0.0",
        last_heartbeat=datetime.utcnow(),
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


@router.get("", response_model=List[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    """List all registered Gaffer cameras."""
    return db.query(Camera).all()


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.put("/{camera_id}/calibration", response_model=CameraResponse)
def set_camera_calibration(camera_id: str, payload: CameraCalibrationRequest, db: Session = Depends(get_db)):
    """
    Operator marks the 4 pitch corners (TL, TR, BR, BL) on a reference frame
    from this fixed camera, once per install. Powers homography-corrected
    distance/speed/xG instead of the flat pixel-to-meter approximation - see
    PitchCalibration.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        PitchCalibration(payload.points, pitch_length_m=40.0, pitch_width_m=20.0)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid calibration points: {e}")

    camera.calibration_points = [list(p) for p in payload.points]
    db.commit()
    db.refresh(camera)
    return camera


@router.post("/{camera_id}/heartbeat", response_model=CameraResponse)
def camera_heartbeat(camera_id: str, payload: CameraHeartbeatRequest, db: Session = Depends(get_db)):
    """Keepalive ping from camera daemon."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera.last_heartbeat = datetime.utcnow()
    if payload.status:
        try:
            camera.status = CameraStatus(payload.status)
        except ValueError:
            pass
    if payload.ip_address:
        camera.ip_address = payload.ip_address
    if payload.firmware_version:
        camera.firmware_version = payload.firmware_version

    db.commit()
    db.refresh(camera)
    return camera


@router.post("/{camera_id}/start_match", response_model=MatchResponse)
def start_match(camera_id: str, payload: StartMatchRequest, db: Session = Depends(get_db)):
    """
    Operator or mobile app commands the camera to start recording a match.
    Creates a Match record in RECORDING status.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Create new match record
    match = Match(
        camera_id=camera.id,
        turf_id=payload.turf_id or camera.turf_id,
        title=payload.title,
        status=MatchStatus.RECORDING,
        start_time=datetime.utcnow(),
        meta_data=payload.meta_data or {},
    )
    camera.status = CameraStatus.RECORDING

    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@router.get("/{camera_id}/next_match")
def get_next_match(camera_id: str, db: Session = Depends(get_db)):
    """
    Camera daemon polling endpoint: checks if there is an active recording scheduled.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    active_match = db.query(Match).filter(
        Match.camera_id == camera.id,
        Match.status == MatchStatus.RECORDING
    ).order_by(Match.created_at.desc()).first()

    if not active_match:
        return {"has_active_match": False, "match": None}

    return {
        "has_active_match": True,
        "match": {
            "id": active_match.id,
            "title": active_match.title,
            "start_time": active_match.start_time.isoformat() if active_match.start_time else None,
        }
    }


@router.post("/{camera_id}/upload_complete")
def upload_complete(camera_id: str, payload: UploadCompleteRequest, db: Session = Depends(get_db)):
    """
    Camera daemon webhook: Signals that local MP4 recording has been uploaded to S3.
    Updates match status to PROCESSING and automatically triggers Celery CV pipeline.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    match = db.query(Match).filter(Match.id == payload.match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    match.video_s3_url = payload.s3_url
    if payload.duration_sec:
        match.duration_sec = payload.duration_sec
    match.status = MatchStatus.PROCESSING
    camera.status = CameraStatus.ONLINE

    db.commit()
    db.refresh(match)

    # Launch Celery CV pipeline asynchronously
    try:
        task_id = trigger_match_pipeline(match.id, payload.s3_url, run_async=True)
    except Exception as e:
        # Fallback to sync execution in environments without a running celery broker
        task_id = trigger_match_pipeline(match.id, payload.s3_url, run_async=False)

    return {
        "ok": True,
        "match_id": match.id,
        "status": match.status,
        "task_id": str(task_id),
        "message": "Video upload acknowledged. Analysis pipeline started.",
    }
