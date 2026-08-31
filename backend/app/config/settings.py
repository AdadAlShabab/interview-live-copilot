"""
Interview Copilot — Application Settings
Reads configuration from environment variables and .env file.
Never hard-codes secrets or model names.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini AI ──────────────────────────────────────────────────────────
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key from https://aistudio.google.com/",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name — configurable without code changes.",
    )

    # ── Free-Tier Protection ───────────────────────────────────────────────
    daily_request_limit: int = Field(
        default=1400,
        description="Max Gemini requests per day before soft cutoff.",
        ge=1,
    )
    session_request_limit: int = Field(
        default=100,
        description="Max Gemini requests per interview session.",
        ge=1,
    )
    warn_at_percent: int = Field(
        default=80,
        description="Warn user when usage reaches this % of the limit.",
        ge=1,
        le=99,
    )

    # ── Transcription ──────────────────────────────────────────────────────
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v2"] = Field(
        default="small",
        description="faster-whisper model size. 'small' recommended for CPU.",
    )
    whisper_device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Compute device for faster-whisper.",
    )
    whisper_threads: int = Field(
        default=4,
        description="CPU threads for faster-whisper inference.",
        ge=1,
    )

    # ── Storage ────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for all local data (SQLite, FAISS, logs).",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/interview_copilot.db",
        description="SQLAlchemy database URL.",
    )
    resume_path: Path = Field(
        default=Path("./data/resume.md"),
        description="Persistent local Markdown resume used by every run.",
    )

    # ── Backend Server ─────────────────────────────────────────────────────
    backend_host: str = Field(default="127.0.0.1")
    backend_port: int = Field(default=8000, ge=1024, le=65535)
    auto_open_browser: bool = Field(
        default=True,
        description="Open browser automatically on server start.",
    )

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_file: Path = Field(default=Path("./data/logs/app.log"))

    # ── Derived / Computed ─────────────────────────────────────────────────
    @property
    def faiss_index_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.log_file.parent

    @property
    def base_url(self) -> str:
        return f"http://{self.backend_host}:{self.backend_port}"

    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Warn if the placeholder value is still set."""
        if v == "your_gemini_api_key_here":
            return ""  # Treat placeholder as missing
        return v

    def is_gemini_configured(self) -> bool:
        """Return True only when a real API key is present."""
        return bool(self.gemini_api_key and len(self.gemini_api_key) > 10)

    def ensure_directories(self) -> None:
        """Create all required local directories."""
        for directory in [
            self.data_dir,
            self.faiss_index_dir,
            self.uploads_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    Uses lru_cache so the .env file is only parsed once at startup.
    """
    return Settings()
