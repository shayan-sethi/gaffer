from celery import Celery
from app.config import settings

celery_app = Celery(
    "gaffer_pipeline",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=[
        "app.tasks.pipeline",
        "app.tasks.track",
        "app.tasks.events",
        "app.tasks.stats",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max for match processing
)
