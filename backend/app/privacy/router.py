"""Privacy and local data-management API."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.privacy.manager import delete_candidate_data
from app.storage.database import get_db_session

router = APIRouter(prefix="/api/privacy", tags=["Privacy"])


@router.delete("/candidate/{candidate_id}")
async def delete_candidate(candidate_id: int, db: AsyncSession = Depends(get_db_session)):
    if not await delete_candidate_data(db, candidate_id):
        raise HTTPException(status_code=404, detail="Candidate not found.")
    return {"status": "deleted", "candidate_id": candidate_id}