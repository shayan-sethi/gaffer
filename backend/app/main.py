import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.init_db import init_db
from app.api.health import router as health_router
from app.api.cameras import router as cameras_router
from app.api.matches import router as matches_router
from app.api.events import router as events_router
from app.api.stats import router as stats_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gaffer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    logger.info("Starting Gaffer Sports Intelligence API...")
    try:
        init_db()
        logger.info("Database schema validated/created.")
    except Exception as e:
        logger.warning(f"Database initialization deferred (database may not be ready yet): {e}")
    yield
    # Shutdown
    logger.info("Shutting down Gaffer API.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Automated Football Computer Vision & Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS middleware for Web Dashboard & Mobile Apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc) if settings.DEBUG else "An unexpected error occurred."},
    )


# Mount routers
app.include_router(health_router)
app.include_router(cameras_router, prefix=settings.API_V1_STR)
app.include_router(matches_router, prefix=settings.API_V1_STR)
app.include_router(events_router, prefix=settings.API_V1_STR)
app.include_router(stats_router, prefix=settings.API_V1_STR)

# Barebones match-visualization dashboard (static HTML/JS, no build step)
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")


@app.get("/")
def root():
    return {
        "name": settings.PROJECT_NAME,
        "status": "online",
        "ui": "/ui",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }
