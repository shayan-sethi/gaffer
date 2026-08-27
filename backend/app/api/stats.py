from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Match, PlayerStat, Player, Event
from app.schemas.stat import MatchStatsOverviewResponse, PlayerStatResponse

router = APIRouter(prefix="/matches/{match_id}/stats", tags=["Stats"])


@router.get("", response_model=MatchStatsOverviewResponse)
def get_match_stats(match_id: str, db: Session = Depends(get_db)):
    """
    Retrieve comprehensive match statistics, team score summary, and all player heatmaps.
    """
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    stats = db.query(PlayerStat).filter(PlayerStat.match_id == match_id).all()
    events = db.query(Event).filter(Event.match_id == match_id).all()

    # Calculate scores
    home_score = sum(1 for e in events if e.event_type == "GOAL" and e.team == "HOME")
    away_score = sum(1 for e in events if e.event_type == "GOAL" and e.team == "AWAY")

    home_xg = round(sum((e.data_json or {}).get("xg", 0.0) for e in events if e.event_type == "SHOT" and e.team == "HOME"), 3)
    away_xg = round(sum((e.data_json or {}).get("xg", 0.0) for e in events if e.event_type == "SHOT" and e.team == "AWAY"), 3)
    home_corners = sum(1 for e in events if e.event_type == "CORNER" and e.team == "HOME")
    away_corners = sum(1 for e in events if e.event_type == "CORNER" and e.team == "AWAY")

    # Aggregate distances, passes, shots
    total_distance = sum(s.distance_m for s in stats)
    total_shots = sum(s.shots for s in stats)
    total_passes = sum(s.passes for s in stats)

    # Team possession - actual time-in-possession (see EventDetector.possession_frames),
    # stored on the match at aggregation time. Falls back to the old pass-count-ratio
    # proxy only for matches processed before this was tracked.
    stored_possession = (match.meta_data or {}).get("possession_pct")
    if stored_possession:
        home_possession = stored_possession["home"]
        away_possession = stored_possession["away"]
    else:
        home_passes = sum(s.passes for s in stats if s.player and s.player.team == "HOME")
        away_passes = sum(s.passes for s in stats if s.player and s.player.team == "AWAY")
        total_p = home_passes + away_passes
        home_possession = round((home_passes / total_p) * 100, 1) if total_p > 0 else 50.0
        away_possession = round(100.0 - home_possession, 1)

    player_stat_responses: List[PlayerStatResponse] = []
    for s in stats:
        player_stat_responses.append(
            PlayerStatResponse(
                id=s.id,
                match_id=s.match_id,
                player_id=s.player_id,
                jersey_number=s.player.jersey_number if s.player else None,
                player_name=s.player.player_name if s.player else None,
                team=s.player.team if s.player else None,
                distance_m=round(s.distance_m, 2),
                top_speed_kmh=round(s.top_speed_kmh, 2),
                avg_speed_kmh=round(s.avg_speed_kmh, 2),
                sprints=s.sprints,
                passes=s.passes,
                shots=s.shots,
                possession_pct=round(s.possession_pct, 1),
                heatmap_json=s.heatmap_json or [],
                updated_at=s.updated_at,
            )
        )

    return MatchStatsOverviewResponse(
        match_id=match.id,
        duration_sec=match.duration_sec,
        home_score=home_score,
        away_score=away_score,
        home_possession_pct=home_possession,
        away_possession_pct=away_possession,
        total_distance_m=round(total_distance, 2),
        total_shots=total_shots,
        total_passes=total_passes,
        zone_touches=(match.meta_data or {}).get("zone_touches"),
        home_xg=home_xg,
        away_xg=away_xg,
        home_corners=home_corners,
        away_corners=away_corners,
        player_stats=player_stat_responses,
    )


@router.get("/players/{player_id}", response_model=PlayerStatResponse)
def get_player_stat(match_id: str, player_id: str, db: Session = Depends(get_db)):
    """Retrieve performance statistics and 2D spatial heatmap for a single player."""
    stat = db.query(PlayerStat).filter(
        PlayerStat.match_id == match_id,
        PlayerStat.player_id == player_id
    ).first()

    if not stat:
        raise HTTPException(status_code=404, detail="Player stats not found for this match")

    return PlayerStatResponse(
        id=stat.id,
        match_id=stat.match_id,
        player_id=stat.player_id,
        jersey_number=stat.player.jersey_number if stat.player else None,
        player_name=stat.player.player_name if stat.player else None,
        team=stat.player.team if stat.player else None,
        distance_m=round(stat.distance_m, 2),
        top_speed_kmh=round(stat.top_speed_kmh, 2),
        avg_speed_kmh=round(stat.avg_speed_kmh, 2),
        sprints=stat.sprints,
        passes=stat.passes,
        shots=stat.shots,
        possession_pct=round(stat.possession_pct, 1),
        heatmap_json=stat.heatmap_json or [],
        updated_at=stat.updated_at,
    )
