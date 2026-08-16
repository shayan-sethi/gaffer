import logging
from celery import chain
from app.tasks.celery_app import celery_app
from app.tasks.track import track_players_task
from app.tasks.events import detect_events_task
from app.tasks.stats import aggregate_stats_task

logger = logging.getLogger(__name__)


def trigger_match_pipeline(match_id: str, s3_url: str = None, run_async: bool = True):
    """
    Triggers the end-to-end CV processing pipeline for a match:
    1. track_players_task (YOLOv8 + Centroid tracking)
    2. detect_events_task (Goals, passes, possession changes)
    3. aggregate_stats_task (Heatmaps, speed, distance, match summary)
    """
    logger.info(f"Triggering match pipeline for match_id={match_id} (async={run_async})")

    if run_async:
        # Build Celery task chain
        pipeline = chain(
            track_players_task.s(match_id, s3_url),
            detect_events_task.s(),
            aggregate_stats_task.s(),
        )
        task_result = pipeline.apply_async()
        logger.info(f"Celery pipeline chain launched with root task ID: {task_result.id}")
        return task_result.id
    else:
        # Run synchronously (useful for integration tests or direct execution)
        tracking_result = track_players_task(match_id, s3_url)
        event_result = detect_events_task(tracking_result)
        stats_result = aggregate_stats_task(event_result)
        return stats_result
