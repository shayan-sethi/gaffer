# Gaffer — Automated Football Turf Intelligence Platform

**Gaffer** is a computer-vision sports intelligence platform that turns fixed football turf cameras into automated match-tracking systems. Turf operators install pre-configured Gaffer cameras; players receive rich match replays, heatmaps, pass analytics, and automated goal clips.

---

## Architecture Overview

```
                                    +----------------------------------------+
                                    |              Gaffer Cloud              |
+----------------------+            |                                        |
|   Gaffer IP Camera   |            |  +----------------+  +--------------+  |
|  (Embedded Linux OS) |  MP4 Clip  |  |  FastAPI API   |  |  PostgreSQL  |  |
|                      |----------->|  | (CRUD, Routes) |  |   Database   |  |
| - H.264 Encoder      | (MinIO/S3) |  +--------+-------+  +-------+------+  |
| - Local SSD Buffer   |            |           |                  |         |
| - Gaffer Daemon      |            |           v                  |         |
| - Chunked Uploader   |  Webhook   |  +----------------+          |         |
| - Auto-pairing QR    |----------->|  | Celery Worker  |          |         |
+----------------------+            |  | (CV Pipeline)  |<---------+         |
                                    |  +--------+-------+                    |
                                    |           |                            |
                                    |           | YOLOv8 Detection           |
                                    |           | Centroid Tracking          |
                                    |           | Goal/Pass Event Engine     |
                                    |           | 2D Spatial Heatmaps        |
                                    |           v                            |
                                    |  +----------------+                    |
                                    |  | Player Replay  |                    |
                                    |  |  & Stats API   |                    |
                                    |  +----------------+                    |
                                    +----------------------------------------+
```

---

## Directory Structure

```
gaffer/
├── backend/                    # Python FastAPI + ML CV pipeline
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint, CORS, lifespan
│   │   ├── config.py          # Pydantic Settings (Postgres, Redis, S3/MinIO)
│   │   ├── db/
│   │   │   ├── database.py    # SQLAlchemy session & engine
│   │   │   ├── models.py      # Camera, Match, Player, Detection, Event, PlayerStat
│   │   │   └── init_db.py     # Schema migration / table creation
│   │   ├── schemas/           # Pydantic validation schemas
│   │   │   ├── camera.py
│   │   │   ├── match.py
│   │   │   ├── event.py
│   │   │   ├── stat.py
│   │   │   └── detection.py
│   │   ├── cv/                # Computer Vision subsystem
│   │   │   ├── detector.py    # YOLOv8 wrapper (players & ball)
│   │   │   ├── tracker.py     # Centroid multi-object tracker & team assignment
│   │   │   ├── events.py      # Rule-based event detector (goals, passes, possession)
│   │   │   └── stats.py       # 2D heatmap matrix, distance (m), top speed (km/h)
│   │   ├── tasks/             # Celery async workers
│   │   │   ├── celery_app.py  # Celery instance configuration
│   │   │   ├── ingest.py      # S3/MinIO download & synthetic clip generator
│   │   │   ├── track.py       # Frame detection & centroid tracking task
│   │   │   ├── events.py      # Event detection task
│   │   │   ├── stats.py       # Stat aggregation & match completion task
│   │   │   └── pipeline.py    # Orchestration task chain
│   │   └── api/               # REST API endpoints
│   │       ├── health.py      # /health, /version
│   │       ├── cameras.py     # /api/cameras (register, start_match, upload_complete)
│   │       ├── matches.py     # /api/matches (CRUD, trigger process)
│   │       ├── events.py      # /api/matches/{id}/events
│   │       └── stats.py       # /api/matches/{id}/stats
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── camera/                     # On-camera daemon (runs on Gaffer hardware)
│   ├── ingest.py              # Main daemon: buffer H.264, upload to S3, notify cloud
│   ├── uploader.py            # Chunked multipart S3/MinIO uploader with retries
│   ├── camera_api.py          # Cloud API client (registration, polling, webhooks)
│   ├── mock_camera.py         # Synthetic camera stream generator for edge testing
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── scripts/
│   └── run_demo_pipeline.py    # End-to-end integration test runner
├── docker-compose.yml         # Local dev: Postgres, Redis, MinIO, API, Celery
└── README.md
```

---

## Quickstart (Local Development)

### 1. Start Dev Infrastructure via Docker Compose
```bash
docker-compose up -d
```
This boots:
- **PostgreSQL 15** on `localhost:5432` (`gaffer:gaffer_dev_password`)
- **Redis 7** on `localhost:6379`
- **MinIO S3 Mock** on `localhost:9000` (Console on `http://localhost:9001`, user/pass: `minioadmin/minioadmin`)
- **Gaffer FastAPI Backend** on `http://localhost:8000`
- **Celery Worker** for CV processing

### 2. Verify System Health
Open your browser or run:
```bash
curl http://localhost:8000/health
```
Interactive Swagger API documentation is available at:
`http://localhost:8000/docs`

---

## Running the End-to-End Pipeline Demo

You can run the full automated CV match tracking demo with a single command:

```bash
python scripts/run_demo_pipeline.py
```

### What this demo does:
1. Creates the database schema and registers a test camera `GAFFER-CAM-DEMO-001`.
2. Schedules a match session.
3. Generates a synthetic 1080p football match clip with moving players and a goal.
4. Executes the **YOLOv8 + Centroid Tracking** pipeline.
5. Runs **Event Detection** (Goals, Passes, Possession Changes).
6. Computes **2D Spatial Heatmaps**, cumulative player distance (meters), and top speed (km/h).
7. Outputs an ASCII visualization of the player heatmap and final scoreline.

---

## Camera Pairing & Lifecycle Flow

```
+----------------+          +-------------------+          +----------------+
|  Turf Operator |          |   Gaffer Camera   |          |  Gaffer Cloud  |
+----------------+          +-------------------+          +----------------+
        |                             |                             |
        |--- 1. Powers on camera ---->|                             |
        |                             |--- 2. POST /cameras/reg --->|
        |                             |<-- Returns Camera API Key --|
        |                                                           |
        |--- 3. Clicks "Start Match" ------------------------------>|
        |                                                           |
        |                             |<-- 4. Polls /next_match ----|
        |                             |                             |
        |                             | [Records H.264 to SSD]      |
        |                             | [Match Finishes]            |
        |                             |                             |
        |                             |--- 5. Uploads MP4 to S3 --->|
        |                             |--- 6. POST /upload_complete>|
        |                                                           |
        |                                                  [Celery CV Chain]
        |                                                  - YOLOv8 detect
        |                                                  - Centroid track
        |                                                  - Goal events
        |                                                  - Player stats
        |                                                           |
        |<-- 7. View Match Replay & Heatmaps (GET /matches/{id}/stats)
```

---

## API Endpoints Reference

### Camera Management
- `POST /api/cameras/register` — Register/pair camera hardware
- `GET /api/cameras` — List cameras
- `POST /api/cameras/{id}/heartbeat` — Keepalive ping
- `POST /api/cameras/{id}/start_match` — Start recording a match session
- `GET /api/cameras/{id}/next_match` — Daemon match polling
- `POST /api/cameras/{id}/upload_complete` — Camera video upload webhook

### Match Management & Processing
- `GET /api/matches` — List all matches
- `POST /api/matches` — Create a match
- `GET /api/matches/{id}` — Get match details and player roster
- `POST /api/matches/{id}/process` — Trigger CV processing pipeline
- `GET /api/matches/{id}/detections` — Paginated frame-by-frame detections

### Events & Statistics
- `GET /api/matches/{id}/events` — Chronological timeline of goals, shots, and passes
- `GET /api/matches/{id}/stats` — Match overview stats, possession, and all player 2D heatmaps
- `GET /api/matches/{id}/stats/players/{player_id}` — Single player performance breakdown
