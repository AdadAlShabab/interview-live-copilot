"""Job-description analysis API."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.job.analyzer import JobAnalyzer
from app.gemini.client import GeminiClient
from app.gemini.schemas import JobAnalysis
from app.storage.database import get_db_session
from app.storage.models import Candidate, JobDescription

router = APIRouter(prefix="/api/job", tags=["Job"])


def get_job_gemini_client() -> GeminiClient:
    """Resolve the shared client without importing the app during module load."""
    from app.main import get_gemini_client

    return get_gemini_client()


class JobAnalysisRequest(BaseModel):
    """Input required to analyze a job description."""

    job_description: str = Field(min_length=1, max_length=100_000)
    candidate_id: int | None = Field(default=None, ge=1)


@router.post("/analyze", response_model=JobAnalysis)
async def analyze_job(
    request: JobAnalysisRequest,
    db: AsyncSession = Depends(get_db_session),
    gemini_client: GeminiClient = Depends(get_job_gemini_client),
) -> JobAnalysis:
    """Analyze a job description against an optional saved candidate profile."""
    candidate = None
    if request.candidate_id is not None:
        result = await db.execute(
            select(Candidate)
            .options(selectinload(Candidate.experiences))
            .where(Candidate.id == request.candidate_id)
        )
        candidate = result.scalar_one_or_none()
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found.")

    try:
        analysis = await JobAnalyzer(gemini_client).analyze(
            request.job_description,
            candidate,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Job analysis failed: {exc}") from exc

    if not isinstance(analysis, JobAnalysis):
        raise HTTPException(status_code=502, detail="Gemini returned an invalid job analysis.")

    record = JobDescription(
        title=analysis.job_title,
        company=analysis.company,
        domain=analysis.domain,
        raw_text=request.job_description.strip(),
        required_skills=analysis.required_skills,
        preferred_skills=analysis.preferred_skills,
        responsibilities=analysis.responsibilities,
        technical_requirements=analysis.technical_requirements,
        behavioral_competencies=analysis.behavioral_competencies,
        likely_questions=analysis.likely_interview_questions,
        important_keywords=analysis.important_keywords,
        skill_matches=[match.model_dump(mode="json") for match in analysis.skill_matches],
    )
    db.add(record)
    await db.commit()

    return analysis