"""
Interview Copilot — Resume API Router

Endpoints for uploading and parsing resumes.
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config.settings import get_settings
from app.gemini.client import GeminiClient
from app.main import get_gemini_client
from app.resume.extractor import ResumeExtractionError, ResumeExtractor
from app.storage.database import get_db_session
from app.storage.models import (
    BehavioralStory,
    Candidate,
    Education,
    Experience,
)
from app.gemini.schemas import CandidateProfile, EducationEntry, ExperienceEntry, BehavioralStory as ProfileStory

router = APIRouter(prefix="/api/resume", tags=["Resume"])
settings = get_settings()


def get_optional_gemini_client() -> GeminiClient | None:
    try:
        return get_gemini_client()
    except RuntimeError:
        return None


def _resume_source_path() -> Path:
    if settings.resume_path.exists():
        return settings.resume_path
    return Path(__file__).resolve().parents[3] / "data" / "resume.md"


def _profile_from_candidate(candidate: Candidate) -> CandidateProfile:
    return CandidateProfile(
        name=candidate.name,
        current_role=candidate.current_role,
        years_experience=candidate.years_experience,
        location=candidate.location,
        summary=candidate.summary,
        industries=candidate.industries or [],
        skills=candidate.skills or [],
        certifications=candidate.certifications or [],
        languages=candidate.languages or [],
        education=[EducationEntry(
            institution=item.institution, degree=item.degree,
            field_of_study=item.field_of_study, graduation_year=item.graduation_year, gpa=item.gpa,
        ) for item in candidate.education],
        experience=[ExperienceEntry(
            company=item.company, role=item.role, start_date=item.start_date, end_date=item.end_date,
            location=item.location, responsibilities=item.responsibilities or [],
            technologies=item.technologies or [], achievements=item.achievements or [], projects=item.projects or [],
        ) for item in candidate.experiences],
        behavioral_stories=[ProfileStory(
            title=item.title, theme=item.theme, situation=item.situation, task=item.task,
            action=item.action, result=item.result, keywords=item.keywords or [],
        ) for item in candidate.stories],
    )


@router.get("/current")
async def current_resume(
    db: AsyncSession = Depends(get_db_session),
    gemini_client: GeminiClient | None = Depends(get_optional_gemini_client),
):
    """Return the persistent local resume profile, extracting it once if needed."""
    result = await db.execute(
        select(Candidate)
        .options(selectinload(Candidate.education), selectinload(Candidate.experiences), selectinload(Candidate.stories))
        .order_by(Candidate.created_at.desc())
        .limit(1)
    )
    candidate = result.scalar_one_or_none()
    if candidate is None:
        resume_path = _resume_source_path()
        if not resume_path.exists():
            raise HTTPException(status_code=404, detail="Persistent resume file not found.")
        if gemini_client is None:
            raise HTTPException(
                status_code=503,
                detail="Gemini is not configured; set GEMINI_API_KEY to extract the local resume.",
            )
        profile = await ResumeExtractor(gemini_client).process_file(resume_path, mime_type="text/markdown")
        candidate = Candidate(
            name=profile.name, current_role=profile.current_role, years_experience=profile.years_experience,
            location=profile.location, summary=profile.summary, industries=profile.industries,
            skills=profile.skills, certifications=profile.certifications, languages=profile.languages,
        )
        db.add(candidate)
        await db.flush()
        for item in profile.education:
            db.add(Education(candidate_id=candidate.id, institution=item.institution, degree=item.degree,
                             field_of_study=item.field_of_study, graduation_year=item.graduation_year, gpa=item.gpa))
        for item in profile.experience:
            db.add(Experience(candidate_id=candidate.id, company=item.company, role=item.role,
                              start_date=item.start_date, end_date=item.end_date, location=item.location,
                              responsibilities=item.responsibilities, technologies=item.technologies,
                              achievements=item.achievements, projects=item.projects))
        for item in profile.behavioral_stories:
            db.add(BehavioralStory(candidate_id=candidate.id, title=item.title, theme=item.theme,
                                   situation=item.situation, task=item.task, action=item.action,
                                   result=item.result, keywords=item.keywords))
        await db.commit()
        return {"candidate_id": candidate.id, "profile": profile.model_dump()}
    return {"candidate_id": candidate.id, "profile": _profile_from_candidate(candidate).model_dump()}


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    gemini_client: GeminiClient = Depends(get_gemini_client),
):
    """
    Upload a resume file, extract text, parse with Gemini,
    and save the structured profile to the database.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # 1. Save uploaded file to disk
    uploads_dir = settings.uploads_dir
    safe_filename = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
    file_path = uploads_dir / safe_filename
    
    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 2. Extract and parse with Gemini
    extractor = ResumeExtractor(gemini_client)
    try:
        profile = await extractor.process_file(file_path, mime_type=file.content_type or "")
    except ResumeExtractionError as e:
        # Cleanup file if parsing failed completely
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error during extraction: {e}")

    # 3. Save to database
    candidate = Candidate(
        name=profile.name,
        current_role=profile.current_role,
        years_experience=profile.years_experience,
        location=profile.location,
        summary=profile.summary,
        industries=profile.industries,
        skills=profile.skills,
        certifications=profile.certifications,
        languages=profile.languages,
    )
    
    db.add(candidate)
    await db.flush()  # To get the candidate.id

    # Add Education
    for edu in profile.education:
        db.add(Education(
            candidate_id=candidate.id,
            institution=edu.institution,
            degree=edu.degree,
            field_of_study=edu.field_of_study,
            graduation_year=edu.graduation_year,
            gpa=edu.gpa,
        ))

    # Add Experience
    for exp in profile.experience:
        db.add(Experience(
            candidate_id=candidate.id,
            company=exp.company,
            role=exp.role,
            start_date=exp.start_date,
            end_date=exp.end_date,
            location=exp.location,
            responsibilities=exp.responsibilities,
            technologies=exp.technologies,
            achievements=exp.achievements,
            projects=exp.projects,
        ))

    # Add Stories
    for story in profile.behavioral_stories:
        db.add(BehavioralStory(
            candidate_id=candidate.id,
            title=story.title,
            theme=story.theme,
            situation=story.situation,
            task=story.task,
            action=story.action,
            result=story.result,
            keywords=story.keywords,
        ))

    await db.commit()
    await db.refresh(candidate)

    # Clean up uploaded file? Depends on privacy settings.
    # We will leave it in `uploads/` for now, privacy settings can delete later.

    return {
        "status": "success",
        "candidate_id": candidate.id,
        "profile": profile.model_dump(),
    }

