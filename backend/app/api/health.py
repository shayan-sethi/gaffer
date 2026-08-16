from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import redis

from app.db.database import get_db
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive system health check for API, PostgreSQL, and Redis.
    """
    # 1. Check PostgreSQL
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # 2. Check Redis
    redis_status = "healthy"
    try:
        r = redis.Redis.from_url(settings.broker_url, socket_timeout=2)
        r.ping()
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"

    status = "ok" if db_status == "healthy" and redis_status == "healthy" else "degraded"

    return {
        "status": status,
        "service": "gaffer-backend",
        "version": "1.0.0",
        "database": db_status,
        "redis": redis_status,
    }


@router.get("/version")
def version():
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "yolo_model": settings.YOLO_MODEL_PATH,
    }
