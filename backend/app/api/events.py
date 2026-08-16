from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Match, Event
from app.schemas.event import EventResponse

router = APIRouter(prefix="/matches/{match_id}/events", tags=["Events"])


@router.get("", response_model=List[EventResponse])
def get_match_events(
    match_id: str,
    event_type: Optional[str] = Query(None, description="Filter by event type (GOAL, SHOT, PASS, POSSESSION_CHANGE)"),
    team: Optional[str] = Query(None, description="Filter by team (HOME, AWAY)"),
    db: Session = Depends(get_db),
):
    """
    Retrieve chronological match event timeline including goals, shots, and passes.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    query = db.query(Event).filter(Event.match_id == match_id)
    if event_type:
        query = query.filter(Event.event_type == event_type.upper())
    if team:
        query = query.filter(Event.team == team.upper())

    return query.order_by(Event.frame_no.asc()).all()
