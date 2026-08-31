"""Tests for job-description analysis."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.gemini.client import GeminiClient
from app.gemini.schemas import JobAnalysis, MatchStrength, SkillMatch
from app.job.router import analyze_job
from app.job.router import JobAnalysisRequest
from app.storage.models import Candidate, Experience


def test_candidate_summary_is_used_in_prompt():
    candidate = Candidate(
        name="Jane Doe",
        current_role="Backend Engineer",
        years_experience=4,
        skills=["Python", "PostgreSQL"],
        experiences=[Experience(company="Acme", role="Engineer")],
    )
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock(
        return_value=JobAnalysis(job_title="Backend Engineer")
    )
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = candidate
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()

    asyncio.run(analyze_job(
        JobAnalysisRequest(job_description="Build Python APIs", candidate_id=1),
        db=session,
        gemini_client=client,
    ))

    prompt = client.generate.call_args.kwargs["prompt"]
    assert "Jane Doe" in prompt
    assert "Python, PostgreSQL" in prompt
    assert "Acme" in prompt
    assert session.commit.await_count == 1


def test_analysis_persists_skill_matches():
    analysis = JobAnalysis(
        job_title="Data Engineer",
        skill_matches=[
            SkillMatch(skill="SQL", strength=MatchStrength.STRONG, evidence="PostgreSQL")
        ],
    )
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock(return_value=analysis)
    session = MagicMock()
    session.get = AsyncMock(return_value=None)
    session.commit = AsyncMock()

    asyncio.run(analyze_job(
        JobAnalysisRequest(job_description="Use SQL"),
        db=session,
        gemini_client=client,
    ))

    record = session.add.call_args.args[0]
    assert record.title == "Data Engineer"
    assert record.skill_matches[0]["strength"] == "strong"