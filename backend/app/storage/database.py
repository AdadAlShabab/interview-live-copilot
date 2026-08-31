"""
Interview Copilot — Database Connection

SQLAlchemy async setup for local SQLite database.
Stores candidate profiles, job analysis, and interview history.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config.settings import get_settings

settings = get_settings()


def _async_database_url(database_url: str) -> str:
    """Use the async SQLite driver even when a plain SQLite URL is configured."""
    if database_url.startswith("sqlite:///") and not database_url.startswith(
        "sqlite+aiosqlite:///"
    ):
        return database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    return database_url

# Create async engine for SQLite
# check_same_thread=False is needed for FastAPI/asyncio with SQLite
engine = create_async_engine(
    _async_database_url(settings.database_url),
    echo=False,  # Set to True for SQL query logging
    connect_args={"check_same_thread": False},
)

# Async session factory
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# Base class for ORM models
Base = declarative_base()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    Yields an active session and closes it after the request.
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI requests
    (e.g., background tasks, initialization).
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
