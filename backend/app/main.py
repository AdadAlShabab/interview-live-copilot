"""
Interview Copilot — FastAPI Application Entry Point

Mounts all routers, initializes shared services, and provides:
- GET /health     — Gemini connectivity + system status
- GET /api/usage  — Current usage stats for the UI
- WebSocket endpoints (added in later phases)

Run with:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import get_settings
from app.gemini.client import GeminiClient, GeminiAuthError
from app.gemini.usage_tracker import UsageTracker

# ── Logging setup ─────────────────────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("interview_copilot")


# ── Shared service singletons (initialized at startup) ────────────────────────
_usage_tracker: UsageTracker | None = None
_gemini_client: GeminiClient | None = None


def get_usage_tracker() -> UsageTracker:
    """FastAPI dependency: return the shared UsageTracker."""
    if _usage_tracker is None:
        raise RuntimeError("UsageTracker not initialized. Did lifespan run?")
    return _usage_tracker


def get_gemini_client() -> GeminiClient:
    """FastAPI dependency: return the shared GeminiClient."""
    if _gemini_client is None:
        raise RuntimeError("GeminiClient not initialized. Did lifespan run?")
    return _gemini_client


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and clean up on shutdown."""
    global _usage_tracker, _gemini_client

    # 1 — Ensure all required directories exist
    settings.ensure_directories()

    # 2 — Setup file logging handler now that log dir exists
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Interview Copilot starting up")
    logger.info(f"  Model: {settings.gemini_model}")
    logger.info(f"  Data dir: {settings.data_dir.resolve()}")
    logger.info(f"  Whisper: {settings.whisper_model} on {settings.whisper_device}")
    logger.info("=" * 60)

    # 3 — Initialize SQLite database tables
    from app.storage.database import init_db
    try:
        await init_db()
        logger.info("Database tables initialized.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # 4 — Initialize UsageTracker
    _usage_tracker = UsageTracker(
        data_dir=settings.data_dir,
        daily_limit=settings.daily_request_limit,
        session_limit=settings.session_request_limit,
        warn_percent=settings.warn_at_percent,
    )

    # 5 — Initialize GeminiClient (may raise GeminiAuthError if key missing)
    if settings.is_gemini_configured():
        try:
            _gemini_client = GeminiClient.from_settings(usage_tracker=_usage_tracker)
            logger.info("Gemini client initialized successfully.")
        except GeminiAuthError as e:
            logger.error(f"Gemini auth error: {e}")
            _gemini_client = None
    else:
        logger.warning(
            "GEMINI_API_KEY not set. Running in limited mode. "
            "Set the key in backend/.env to enable AI features."
        )
        _gemini_client = None

    yield  # Application runs here

    # Shutdown cleanup
    if _usage_tracker and _usage_tracker._current_session:
        _usage_tracker.end_session()
    logger.info("Interview Copilot shut down cleanly.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interview Copilot API",
    description="Real-time AI interview assistant powered by Google Gemini",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend (PyWebView/browser) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount frontend static files (added in Phase 9) ────────────────────────────
_frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ── Routes ────────────────────────────────────────────────────────────────────

from app.resume.router import router as resume_router
app.include_router(resume_router)
from app.job.router import router as job_router
app.include_router(job_router)
from app.interview.router import router as interview_router
app.include_router(interview_router)
from app.privacy.router import router as privacy_router
app.include_router(privacy_router)

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint — verifies:
    - Application is running
    - Gemini API connectivity
    - Configuration status
    - Usage tracker status
    """
    tracker = _usage_tracker
    client = _gemini_client

    # Gemini status
    if client is None:
        gemini_status = {
            "connected": False,
            "model": settings.gemini_model,
            "error": "API key not configured" if not settings.is_gemini_configured()
                     else "Client initialization failed",
        }
    else:
        gemini_status = await client.health_check()

    # Usage status
    usage_status = tracker.get_status() if tracker else {}

    return JSONResponse(
        content={
            "status": "ok",
            "version": "0.1.0",
            "gemini": gemini_status,
            "usage": usage_status,
            "config": {
                "model": settings.gemini_model,
                "whisper_model": settings.whisper_model,
                "whisper_device": settings.whisper_device,
                "daily_request_limit": settings.daily_request_limit,
            },
        }
    )


@app.get("/api/usage", tags=["System"])
async def get_usage():
    """Return current API usage stats for the UI status bar."""
    tracker = _usage_tracker
    if tracker is None:
        return JSONResponse(content={"error": "tracker_not_initialized"}, status_code=503)
    return JSONResponse(content=tracker.get_status())


@app.post("/api/session/start", tags=["Session"])
async def start_session():
    """Begin a new interview session — resets per-session counters."""
    tracker = _usage_tracker
    if tracker is None:
        return JSONResponse(content={"error": "tracker_not_initialized"}, status_code=503)
    session_id = str(uuid.uuid4())
    tracker.start_session(session_id)
    return JSONResponse(content={"session_id": session_id, "status": "started"})


@app.post("/api/session/end", tags=["Session"])
async def end_session():
    """End the current interview session and persist usage totals."""
    tracker = _usage_tracker
    if tracker is None:
        return JSONResponse(content={"error": "tracker_not_initialized"}, status_code=503)
    session = tracker.end_session()
    if session:
        return JSONResponse(content={
            "session_id": session.session_id,
            "requests": session.requests,
            "total_tokens": session.total_tokens,
            "errors": session.errors,
        })
    return JSONResponse(content={"status": "no_active_session"})


@app.get("/", tags=["System"])
async def root():
    """Redirect info for root path."""
    return {
        "app": "Interview Copilot",
        "version": "0.1.0",
        "health": "/health",
        "docs": "/docs",
        "ui": "/ui" if _frontend_dir.exists() else "Frontend not yet built (Phase 9)",
    }
