"""
conftest.py — shared pytest fixtures and configuration.
Sets up a test environment with a temporary .env
so tests don't require a real GEMINI_API_KEY.
"""

import os
import pytest
from pathlib import Path


# Set test environment variables BEFORE any app imports
# This prevents pydantic-settings from reading the real .env
os.environ.setdefault("GEMINI_API_KEY", "test-fake-key-for-unit-tests-1234567890")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
os.environ.setdefault("DAILY_REQUEST_LIMIT", "100")
os.environ.setdefault("SESSION_REQUEST_LIMIT", "20")
os.environ.setdefault("WARN_AT_PERCENT", "80")
os.environ.setdefault("WHISPER_MODEL", "small")
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("AUTO_OPEN_BROWSER", "false")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
