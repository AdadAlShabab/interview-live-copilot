"""
Interview Copilot — Phase 2 Tests: Resume Intelligence

Tests:
1. Extractor handles text extraction correctly
2. Extractor sends correct prompt to Gemini
3. Router handles file upload
4. Router saves structured profile to DB
"""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.gemini.client import GeminiClient
from app.gemini.schemas import CandidateProfile, EducationEntry, ExperienceEntry, BehavioralStory
from app.main import app
from app.resume.extractor import ResumeExtractor


@pytest.fixture
def mock_gemini_client():
    client = MagicMock(spec=GeminiClient)
    # Mock the async generate method
    profile = CandidateProfile(
        name="John Doe",
        current_role="Software Engineer",
        years_experience=5.0,
        location="New York",
        summary="A great software engineer.",
        skills=["Python", "SQL"],
        education=[
            EducationEntry(
                institution="State University",
                degree="BSc Computer Science",
                graduation_year="2018"
            )
        ],
        experience=[
            ExperienceEntry(
                company="Tech Corp",
                role="Backend Developer",
                start_date="Jan 2019",
                end_date="Present",
                responsibilities=["Built APIs"],
            )
        ],
        behavioral_stories=[
            BehavioralStory(
                title="API Migration",
                theme="success",
                situation="Slow APIs",
                task="Migrate to FastAPI",
                action="Rewrote using async/await",
                result="10x speedup"
            )
        ]
    )
    client.generate = AsyncMock(return_value=profile)
    return client


@pytest.fixture
def test_client(mock_gemini_client):
    # Override the dependency for the test
    from app.main import get_gemini_client
    app.dependency_overrides[get_gemini_client] = lambda: mock_gemini_client
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


class TestResumeExtractor:
    @pytest.mark.asyncio
    async def test_txt_extraction_and_parsing(self, tmp_path, mock_gemini_client):
        # Create a fake text resume
        file_path = tmp_path / "resume.txt"
        file_path.write_text("John Doe\nSoftware Engineer\nPython, SQL")

        extractor = ResumeExtractor(mock_gemini_client)
        profile = await extractor.process_file(file_path, mime_type="text/plain")

        assert profile.name == "John Doe"
        assert len(profile.skills) == 2
        mock_gemini_client.generate.assert_called_once()
        kwargs = mock_gemini_client.generate.call_args.kwargs
        assert "John Doe" in kwargs["prompt"]


class TestResumeRouter:
    def test_upload_resume_txt(self, test_client, tmp_path):
        # Create a fake text file in memory to upload
        file_content = b"Fake resume content"
        
        # We need to mock the db session to avoid writing to the real db during unit tests
        # Or let it write to the test.db we configured in conftest.py
        # Since conftest.py points database_url to a test SQLite DB, and we call init_db on lifespan,
        # we can just use the real db for this test.
        
        # But wait, TestClient does not run lifespan by default unless used in a `with TestClient(app)` block,
        # which we do in the fixture. However, lifespan async tasks might need an event loop.
        # Let's just mock the DB dependency.
        
        from app.storage.database import get_db_session
        from sqlalchemy.ext.asyncio import AsyncSession
        mock_session = AsyncMock(spec=AsyncSession)
        app.dependency_overrides[get_db_session] = lambda: mock_session

        response = test_client.post(
            "/api/resume/upload",
            files={"file": ("resume.txt", file_content, "text/plain")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["profile"]["name"] == "John Doe"
        
        # Check that we tried to add to db
        assert mock_session.add.call_count >= 1
        assert mock_session.commit.call_count == 1
