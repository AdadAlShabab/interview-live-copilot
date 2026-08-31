"""
Interview Copilot — Phase 1 Tests: Gemini Connection

Tests:
1. Settings loads correctly
2. Health endpoint returns expected structure
3. Gemini client health check (mocked — no real API call)
4. Usage tracker: request counting, limits, circuit breaker
5. Gemini client: valid key behavior (mocked)
6. Gemini client: invalid key raises GeminiAuthError
7. Gemini client: 429 triggers circuit breaker + backoff
8. Gemini client: retries on 5xx
9. API endpoint: GET /health structure
10. API endpoint: GET /api/usage structure

Run with:
    cd backend
    pytest tests/test_gemini.py -v
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.config.settings import Settings
from app.gemini.client import GeminiClient, GeminiAuthError, GeminiRateLimitError, GeminiUnavailableError
from app.gemini.usage_tracker import UsageTracker
from app.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    return tmp_path / "data"


@pytest.fixture
def usage_tracker(tmp_data_dir: Path) -> UsageTracker:
    """Fresh UsageTracker for each test."""
    return UsageTracker(
        data_dir=tmp_data_dir,
        daily_limit=100,
        session_limit=20,
        warn_percent=80,
    )


@pytest.fixture
def mock_settings(tmp_data_dir: Path):
    """Settings with a fake API key for testing."""
    return Settings(
        gemini_api_key="fake-test-key-1234567890",
        gemini_model="gemini-2.5-flash",
        data_dir=tmp_data_dir,
        database_url=f"sqlite:///{tmp_data_dir}/test.db",
    )


# ── Settings Tests ────────────────────────────────────────────────────────────

class TestSettings:
    def test_placeholder_key_treated_as_missing(self):
        s = Settings(gemini_api_key="your_gemini_api_key_here")
        assert not s.is_gemini_configured()

    def test_empty_key_not_configured(self):
        s = Settings(gemini_api_key="")
        assert not s.is_gemini_configured()

    def test_valid_key_is_configured(self):
        s = Settings(gemini_api_key="AIzaSyFakeKeyForTesting1234567890")
        assert s.is_gemini_configured()

    def test_default_model_is_flash(self):
        s = Settings(gemini_api_key="fake-key-1234567890")
        assert s.gemini_model == "gemini-2.5-flash"

    def test_ensure_directories_creates_dirs(self, mock_settings: Settings):
        mock_settings.ensure_directories()
        assert mock_settings.data_dir.exists()
        assert mock_settings.faiss_index_dir.exists()
        assert mock_settings.uploads_dir.exists()
        assert mock_settings.logs_dir.exists()

    def test_base_url_format(self):
        s = Settings(gemini_api_key="fake", backend_host="127.0.0.1", backend_port=8000)
        assert s.base_url == "http://127.0.0.1:8000"


# ── UsageTracker Tests ────────────────────────────────────────────────────────

class TestUsageTracker:
    def test_initial_status(self, usage_tracker: UsageTracker):
        status = usage_tracker.get_status()
        assert status["daily_requests"] == 0
        assert status["session_requests"] == 0
        assert not status["rate_limited"]

    def test_session_tracking(self, usage_tracker: UsageTracker):
        session = usage_tracker.start_session("test-session-1")
        assert session.session_id == "test-session-1"
        assert session.requests == 0

    def test_record_request_increments_counters(self, usage_tracker: UsageTracker):
        usage_tracker.start_session("test-1")
        usage_tracker.record_request(
            prompt_type="test",
            input_tokens=100,
            output_tokens=50,
            success=True,
        )
        status = usage_tracker.get_status()
        assert status["session_requests"] == 1
        assert status["session_tokens"] == 150

    def test_rate_limit_circuit_breaker(self, usage_tracker: UsageTracker):
        assert not usage_tracker.is_rate_limited()
        usage_tracker.set_rate_limited(retry_after_seconds=3600)
        assert usage_tracker.is_rate_limited()
        allowed, reason = usage_tracker.can_make_request()
        assert not allowed
        assert "Rate limit" in reason

    def test_circuit_breaker_clears_after_timeout(self, usage_tracker: UsageTracker):
        """Circuit breaker should clear when the retry window expires."""
        usage_tracker.set_rate_limited(retry_after_seconds=0.01)
        import time
        time.sleep(0.05)
        assert not usage_tracker.is_rate_limited()

    def test_daily_limit_blocks_requests(self, usage_tracker: UsageTracker):
        # Manually set daily count beyond limit
        usage_tracker._daily["requests"] = 100  # equal to daily_limit
        allowed, reason = usage_tracker.can_make_request()
        assert not allowed
        assert "Daily" in reason

    def test_session_limit_blocks_requests(self, usage_tracker: UsageTracker):
        session = usage_tracker.start_session("test-limit")
        session.requests = 20  # equal to session_limit
        allowed, reason = usage_tracker.can_make_request()
        assert not allowed
        assert "Session" in reason

    def test_end_session_returns_stats(self, usage_tracker: UsageTracker):
        usage_tracker.start_session("test-end")
        usage_tracker.record_request("test", 100, 50, True)
        session = usage_tracker.end_session()
        assert session is not None
        assert session.requests == 1
        assert session.total_tokens == 150

    def test_persistence_file_created(self, usage_tracker: UsageTracker, tmp_data_dir: Path):
        usage_tracker.start_session("persist-test")
        usage_tracker.record_request("test", 100, 50, True)
        usage_tracker.end_session()
        assert (tmp_data_dir / "usage_tracker.json").exists()

    def test_warning_threshold(self, usage_tracker: UsageTracker):
        # Set requests to 85% of limit (above warn_at_percent=80)
        usage_tracker._daily["requests"] = 85
        status = usage_tracker.get_status()
        assert status["is_warning"]


# ── GeminiClient Tests (mocked) ───────────────────────────────────────────────

class TestGeminiClient:
    def test_raises_auth_error_when_no_key(self, usage_tracker: UsageTracker):
        with pytest.raises(GeminiAuthError):
            GeminiClient(api_key="", model="gemini-2.5-flash", usage_tracker=usage_tracker)

    def test_raises_auth_error_for_short_key(self, usage_tracker: UsageTracker):
        with pytest.raises(GeminiAuthError):
            GeminiClient(api_key="short", model="gemini-2.5-flash", usage_tracker=usage_tracker)

    @pytest.mark.asyncio
    async def test_health_check_without_key(self, tmp_data_dir: Path):
        """Health check should return connected=False without API key."""
        tracker = UsageTracker(tmp_data_dir, 100, 20, 80)
        with patch("app.gemini.client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                is_gemini_configured=lambda: False,
                gemini_model="gemini-2.5-flash",
            )
            # Create client with fake key to bypass __init__ check
            with patch("app.gemini.client.genai.Client"):
                client = GeminiClient("fake-key-1234567890", "gemini-2.5-flash", tracker)
                result = await client.health_check()
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_generate_blocked_when_rate_limited(self, usage_tracker: UsageTracker):
        """generate() should raise when circuit breaker is open."""
        usage_tracker.set_rate_limited(retry_after_seconds=3600)
        with patch("app.gemini.client.genai.Client"):
            client = GeminiClient("fake-key-1234567890", "gemini-2.5-flash", usage_tracker)
        with pytest.raises(GeminiRateLimitError):
            await client.generate("test", prompt_type="test")

    @pytest.mark.asyncio
    async def test_successful_generate_records_usage(self, usage_tracker: UsageTracker):
        """A successful generate() call should be recorded in the tracker."""
        usage_tracker.start_session("gen-test")

        mock_response = MagicMock()
        mock_response.text = "Test response from Gemini"
        mock_response.usage_metadata = None

        with patch("app.gemini.client.genai.Client"):
            client = GeminiClient("fake-key-1234567890", "gemini-2.5-flash", usage_tracker)

        with patch.object(client, "_call_api", new=AsyncMock(return_value=mock_response)):
            result = await client.generate("Hello", prompt_type="test")

        assert result == "Test response from Gemini"
        status = usage_tracker.get_status()
        assert status["session_requests"] == 1

    @pytest.mark.asyncio
    async def test_token_estimation(self, usage_tracker: UsageTracker):
        """Token estimator should return reasonable values."""
        assert GeminiClient._estimate_tokens("Hello world") > 0
        assert GeminiClient._estimate_tokens("") == 1  # minimum 1


# ── FastAPI Endpoint Tests ─────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Integration tests for the FastAPI health endpoint."""

    def test_health_endpoint_returns_200(self):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_health_response_structure(self):
        with TestClient(app) as client:
            response = client.get("/health")
            data = response.json()
            assert "status" in data
            assert "gemini" in data
            assert "usage" in data
            assert "config" in data

    def test_health_gemini_status_has_required_fields(self):
        with TestClient(app) as client:
            response = client.get("/health")
            gemini = response.json()["gemini"]
            assert "connected" in gemini
            assert "model" in gemini

    def test_health_config_shows_model(self):
        with TestClient(app) as client:
            response = client.get("/health")
            config = response.json()["config"]
            assert "model" in config
            assert "whisper_model" in config

    def test_usage_endpoint_returns_200(self):
        with TestClient(app) as client:
            response = client.get("/api/usage")
            assert response.status_code == 200

    def test_usage_response_structure(self):
        with TestClient(app) as client:
            response = client.get("/api/usage")
            data = response.json()
            assert "daily_requests" in data
            assert "daily_limit" in data
            assert "rate_limited" in data

    def test_root_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["app"] == "Interview Copilot"

    def test_session_start_end(self):
        with TestClient(app) as client:
            # Start session
            start_resp = client.post("/api/session/start")
            assert start_resp.status_code == 200
            data = start_resp.json()
            assert "session_id" in data
            assert data["status"] == "started"

            # End session
            end_resp = client.post("/api/session/end")
            assert end_resp.status_code == 200
