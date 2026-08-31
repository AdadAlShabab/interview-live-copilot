"""
Interview Copilot — Gemini API Usage Tracker

Tracks requests and estimated token consumption per session and per day.
Persists daily counts to a JSON file so they survive restarts.
Provides rate-limit status, warnings, and circuit-breaker state.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """Single Gemini API request record."""
    timestamp: str
    prompt_type: str          # e.g. "resume_extraction", "answer_generation"
    input_tokens: int
    output_tokens: int
    success: bool
    error_code: Optional[str] = None  # e.g. "429", "timeout"
    latency_ms: Optional[float] = None


@dataclass
class SessionStats:
    """Statistics for a single interview session."""
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0
    rate_limit_hits: int = 0
    records: list[RequestRecord] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Always $0 on free tier — shown as reassurance."""
        return 0.0


class UsageTracker:
    """
    Thread-safe usage tracker for Gemini API.

    Responsibilities:
    - Count requests and tokens per session and per day
    - Persist daily counts across restarts
    - Detect and report rate-limit exhaustion
    - Provide circuit-breaker state for the Gemini client
    - Emit warnings when approaching configured limits
    """

    _instance: Optional["UsageTracker"] = None
    _lock = threading.Lock()

    def __init__(self, data_dir: Path, daily_limit: int, session_limit: int, warn_percent: int):
        self._data_dir = data_dir
        self._daily_limit = daily_limit
        self._session_limit = session_limit
        self._warn_percent = warn_percent
        self._storage_file = data_dir / "usage_tracker.json"

        self._write_lock = threading.Lock()

        # Current session
        self._current_session: Optional[SessionStats] = None

        # Daily persistent counts
        self._daily: dict = self._load_daily()

        # Circuit breaker — set True when 429 received, cleared after backoff
        self._rate_limited: bool = False
        self._rate_limit_until: Optional[datetime] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def start_session(self, session_id: str) -> SessionStats:
        """Begin tracking a new interview session."""
        self._current_session = SessionStats(session_id=session_id)
        logger.info(f"[UsageTracker] Started session: {session_id}")
        return self._current_session

    def end_session(self) -> Optional[SessionStats]:
        """Finalize the current session and persist totals."""
        if self._current_session:
            self._persist_daily(
                requests=self._current_session.requests,
                tokens=self._current_session.total_tokens,
            )
            session = self._current_session
            self._current_session = None
            logger.info(
                f"[UsageTracker] Session ended — "
                f"requests={session.requests}, tokens={session.total_tokens}"
            )
            return session
        return None

    def record_request(
        self,
        prompt_type: str,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error_code: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record a single API call outcome."""
        with self._write_lock:
            record = RequestRecord(
                timestamp=datetime.now().isoformat(),
                prompt_type=prompt_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                error_code=error_code,
                latency_ms=latency_ms,
            )

            if self._current_session:
                self._current_session.requests += 1
                self._current_session.input_tokens += input_tokens
                self._current_session.output_tokens += output_tokens
                self._current_session.records.append(record)
                if not success:
                    self._current_session.errors += 1
                if error_code == "429":
                    self._current_session.rate_limit_hits += 1

        self._check_limits()

    def set_rate_limited(self, retry_after_seconds: float = 60.0) -> None:
        """Mark that the API returned 429 — activate circuit breaker."""
        import time
        from datetime import timedelta

        self._rate_limited = True
        self._rate_limit_until = datetime.now() + timedelta(seconds=retry_after_seconds)
        logger.warning(
            f"[UsageTracker] Rate limited. Circuit breaker open for {retry_after_seconds:.0f}s."
        )

    def clear_rate_limit(self) -> None:
        """Reset circuit breaker after backoff period."""
        self._rate_limited = False
        self._rate_limit_until = None
        logger.info("[UsageTracker] Rate limit cleared. Circuit breaker closed.")

    def is_rate_limited(self) -> bool:
        """Return True if the circuit breaker is open."""
        if not self._rate_limited:
            return False
        if self._rate_limit_until and datetime.now() > self._rate_limit_until:
            self.clear_rate_limit()
            return False
        return True

    def can_make_request(self) -> tuple[bool, str]:
        """
        Return (True, "") if a request is allowed, or
        (False, reason) if it should be blocked.
        """
        if self.is_rate_limited():
            remaining = ""
            if self._rate_limit_until:
                secs = (self._rate_limit_until - datetime.now()).total_seconds()
                remaining = f" Retry in {secs:.0f}s."
            return False, f"Rate limit active.{remaining}"

        daily_requests = self._daily.get("requests", 0)
        if daily_requests >= self._daily_limit:
            return False, (
                f"Daily request limit reached ({daily_requests}/{self._daily_limit}). "
                "Limit resets at midnight Pacific time."
            )

        if self._current_session and self._current_session.requests >= self._session_limit:
            return False, (
                f"Session request limit reached ({self._current_session.requests}/{self._session_limit})."
            )

        return True, ""

    def get_status(self) -> dict:
        """Return a status dict suitable for the UI status bar."""
        daily_requests = self._daily.get("requests", 0)
        daily_tokens = self._daily.get("tokens", 0)
        session = self._current_session

        warn_threshold = int(self._daily_limit * self._warn_percent / 100)
        is_warning = daily_requests >= warn_threshold

        return {
            "rate_limited": self.is_rate_limited(),
            "rate_limit_until": self._rate_limit_until.isoformat() if self._rate_limit_until else None,
            "daily_requests": daily_requests,
            "daily_limit": self._daily_limit,
            "daily_tokens": daily_tokens,
            "is_warning": is_warning,
            "session_requests": session.requests if session else 0,
            "session_tokens": session.total_tokens if session else 0,
            "session_errors": session.errors if session else 0,
            "session_rate_limit_hits": session.rate_limit_hits if session else 0,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _check_limits(self) -> None:
        """Log a warning if approaching configured limits."""
        daily_requests = self._daily.get("requests", 0)
        warn_threshold = int(self._daily_limit * self._warn_percent / 100)
        if daily_requests >= warn_threshold:
            logger.warning(
                f"[UsageTracker] Approaching daily limit: "
                f"{daily_requests}/{self._daily_limit} requests used."
            )

    def _load_daily(self) -> dict:
        """Load persisted daily counters, resetting if it's a new day."""
        today = date.today().isoformat()
        if self._storage_file.exists():
            try:
                data = json.loads(self._storage_file.read_text())
                if data.get("date") == today:
                    logger.debug(f"[UsageTracker] Loaded daily stats for {today}.")
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        # New day or corrupt file — start fresh
        return {"date": today, "requests": 0, "tokens": 0}

    def _persist_daily(self, requests: int, tokens: int) -> None:
        """Add session totals to the daily persistent store."""
        with self._write_lock:
            self._daily["requests"] = self._daily.get("requests", 0) + requests
            self._daily["tokens"] = self._daily.get("tokens", 0) + tokens
            try:
                self._data_dir.mkdir(parents=True, exist_ok=True)
                self._storage_file.write_text(json.dumps(self._daily, indent=2))
            except OSError as e:
                logger.error(f"[UsageTracker] Failed to persist daily stats: {e}")
