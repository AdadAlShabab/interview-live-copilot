"""Interview answer-generation API."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config.settings import get_settings
from app.gemini.client import GeminiClient
from app.gemini.schemas import AnswerResponse
from app.interview.answer_generator import AnswerGenerator
from app.interview.classifier import classify
from app.interview.conversation_memory import ConversationMemory, is_followup
from app.interview.question_detector import is_question
from app.interview.session import InterviewState
from app.storage.database import get_db_session
from app.storage.database import db_context
from app.storage.models import Candidate, InterviewSession, InterviewTurn

router = APIRouter(prefix="/api/interview", tags=["Interview"])


def get_interview_gemini_client() -> GeminiClient:
    from app.main import get_gemini_client

    return get_gemini_client()


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    candidate_id: int = Field(ge=1)
    job_title: str = "Not provided"
    key_requirements: list[str] = Field(default_factory=list)
    recent_context: str = "None"


@router.post("/answer", response_model=AnswerResponse)
async def generate_answer(
    request: AnswerRequest,
    db: AsyncSession = Depends(get_db_session),
    gemini_client: GeminiClient = Depends(get_interview_gemini_client),
) -> AnswerResponse:
    result = await db.execute(
        select(Candidate)
        .options(selectinload(Candidate.experiences), selectinload(Candidate.stories))
        .where(Candidate.id == request.candidate_id)
    )
    candidate = result.scalar_one_or_none()
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    try:
        return await AnswerGenerator(
            gemini_client,
            get_settings().faiss_index_dir,
        ).generate(
            question=request.question,
            candidate=candidate,
            job_title=request.job_title,
            key_requirements=request.key_requirements,
            recent_context=request.recent_context,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc


@router.websocket("/ws/{session_uuid}")
async def interview_websocket(websocket: WebSocket, session_uuid: str):
    """Receive transcript text and return answers for detected questions."""
    await websocket.accept()
    state = InterviewState(session_uuid=session_uuid)
    try:
        setup = await websocket.receive_json()
        candidate_id = int(setup["candidate_id"])
        job_title = str(setup.get("job_title", "Not provided"))
        requirements = [str(item) for item in setup.get("key_requirements", [])]
        from app.main import get_gemini_client

        client = get_gemini_client()
        async with db_context() as db:
            result = await db.execute(
                select(Candidate)
                .options(selectinload(Candidate.experiences), selectinload(Candidate.stories))
                .where(Candidate.id == candidate_id)
            )
            candidate = result.scalar_one_or_none()
            if candidate is None:
                await websocket.send_json({"error": "Candidate not found."})
                return
            db_session = InterviewSession(
                session_uuid=session_uuid,
                candidate_id=candidate_id,
            )
            db.add(db_session)
            await db.flush()
            await websocket.send_json({"type": "ready", "session_uuid": session_uuid})
            generator = AnswerGenerator(client, get_settings().faiss_index_dir)
            memory = ConversationMemory(client)
            while True:
                message = await websocket.receive_json()
                text = str(message.get("text", "")).strip()
                speaker = str(message.get("speaker", "interviewer"))
                if not text:
                    continue
                state.add_turn(speaker, text)
                memory.add_turn(speaker, text)
                turn = InterviewTurn(
                    session_id=db_session.id,
                    turn_index=state.turn_count,
                    speaker=speaker,
                    text=text,
                    is_question=int(is_question(text)),
                )
                db.add(turn)
                question_detected = is_question(text)
                response = {
                    "type": "transcript",
                    "speaker": speaker,
                    "text": text,
                    "is_question": question_detected,
                    "is_followup": question_detected and is_followup(text),
                }
                if question_detected:
                    classification = classify(text)
                    answer = await generator.generate(
                        text,
                        candidate,
                        job_title=job_title,
                        key_requirements=requirements,
                        recent_context=memory.context,
                    )
                    turn.question_type = classification.question_type.value
                    turn.ai_suggested_answer = answer.suggested_answer
                    turn.ai_evidence_used = answer.evidence_used
                    response["classification"] = classification.model_dump(mode="json")
                    response["answer"] = answer.model_dump(mode="json")
                summary = await memory.summarize()
                if summary:
                    db_session.memory_state = summary.model_dump(mode="json")
                    response["summary"] = summary.model_dump(mode="json")
                await db.commit()
                await websocket.send_json(response)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})