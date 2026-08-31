"""Local data deletion controls."""

import shutil

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.storage.models import (
    BehavioralStory,
    Candidate,
    Education,
    Experience,
    InterviewSession,
    InterviewTurn,
)


async def delete_candidate_data(db: AsyncSession, candidate_id: int) -> bool:
    """Delete a candidate and all associated profile records and vector data."""
    candidate = await db.get(Candidate, candidate_id)
    if candidate is None:
        return False
    await db.execute(delete(BehavioralStory).where(BehavioralStory.candidate_id == candidate_id))
    await db.execute(delete(Experience).where(Experience.candidate_id == candidate_id))
    await db.execute(delete(Education).where(Education.candidate_id == candidate_id))
    sessions = await db.scalars(
        select(InterviewSession.id).where(InterviewSession.candidate_id == candidate_id)
    )
    session_ids = list(sessions)
    if session_ids:
        await db.execute(delete(InterviewTurn).where(InterviewTurn.session_id.in_(session_ids)))
        await db.execute(delete(InterviewSession).where(InterviewSession.id.in_(session_ids)))
    await db.execute(delete(Candidate).where(Candidate.id == candidate_id))
    await db.commit()
    shutil.rmtree(get_settings().faiss_index_dir / f"candidate_{candidate_id}", ignore_errors=True)
    return True