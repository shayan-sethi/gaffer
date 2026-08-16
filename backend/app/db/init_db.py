import logging
from app.db.database import engine, Base
from app.db.models import Camera, Match, Player, Detection, Event, PlayerStat  # noqa: F401

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
