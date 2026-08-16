from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Match, MatchStatus, Player, Detection
from app.schemas.match import MatchResponse, MatchDetailResponse, TriggerProcessRequest
from app.schemas.detection import DetectionResponse
from app.tasks.pipeline import trigger_match_pipeline

router = APIRouter(prefix="/matches", tags=["Matches"])


@router.get("", response_model=List[MatchResponse])
def list_matches(
    turf_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List matches with optional status and turf filters."""
    query = db.query(Match)
    if turf_id:
        query = query.filter(Match.turf_id == turf_id)
    if status:
        query = query.filter(Match.status == status)

    return query.order_by(Match.created_at.desc()).offset(offset).limit(limit).all()


@router.post("", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
def create_match(
    title: str = "5-a-side Match",
    turf_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    video_s3_url: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Manually create a match."""
    match = Match(
        title=title,
        turf_id=turf_id,
        camera_id=camera_id,
        video_s3_url=video_s3_url,
        status=MatchStatus.PENDING,
        date=datetime.utcnow(),
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@router.get("/{match_id}", response_model=MatchDetailResponse)
def get_match(match_id: str, db: Session = Depends(get_db)):
    """Retrieve full match metadata and assigned players."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.post("/{match_id}/process")
def trigger_process_match(
    match_id: str,
    payload: Optional[TriggerProcessRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Manually kick off the CV tracking and analytics pipeline for a match.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    video_url = payload.video_s3_url if payload and payload.video_s3_url else match.video_s3_url
    if not video_url:
        video_url = "synthetic"  # Fallback to test video generator

    match.video_s3_url = video_url
    match.status = MatchStatus.PROCESSING
    db.commit()

    try:
        task_id = trigger_match_pipeline(match.id, video_url, run_async=True)
    except Exception:
        task_id = trigger_match_pipeline(match.id, video_url, run_async=False)

    return {
        "ok": True,
        "match_id": match.id,
        "status": "PROCESSING",
        "task_id": str(task_id),
        "message": "Match processing pipeline started.",
    }


@router.get("/{match_id}/detections", response_model=List[DetectionResponse])
def get_match_detections(
    match_id: str,
    frame_no: Optional[int] = Query(None),
    track_id: Optional[int] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve raw computer-vision player/ball detections by frame or track."""
    query = db.query(Detection).filter(Detection.match_id == match_id)
    if frame_no is not None:
        query = query.filter(Detection.frame_no == frame_no)
    if track_id is not None:
        query = query.filter(Detection.track_id == track_id)

    return query.order_by(Detection.frame_no.asc()).offset(offset).limit(limit).all()


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_match(match_id: str, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    db.delete(match)
    db.commit()
    return None
