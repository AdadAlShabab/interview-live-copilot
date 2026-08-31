"""Evidence-grounded interview answer generation."""

from app.gemini.client import GeminiClient
from app.gemini.prompts import ANSWER_GENERATION_SYSTEM, ANSWER_GENERATION_USER
from app.gemini.schemas import AnswerResponse
from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.vector_store import VectorStore, candidate_evidence
from app.storage.models import Candidate


class AnswerGenerator:
    """Retrieve compact candidate evidence and generate a structured answer."""

    def __init__(self, gemini_client: GeminiClient, index_dir):
        self._gemini_client = gemini_client
        self._index_dir = index_dir
        self._store_cache: dict[int, VectorStore] = {}

    def _get_store(self, candidate: Candidate) -> VectorStore:
        store = self._store_cache.get(candidate.id)
        if store is not None:
            return store

        store = VectorStore(self._index_dir / f"candidate_{candidate.id}")
        store.add(candidate_evidence(candidate))
        self._store_cache[candidate.id] = store
        return store

    async def generate(
        self,
        question: str,
        candidate: Candidate,
        job_title: str = "Not provided",
        key_requirements: list[str] | None = None,
        recent_context: str = "None",
    ) -> AnswerResponse:
        store = self._get_store(candidate)
        evidence = EvidenceRetriever(store).retrieve(question, limit=3)
        evidence_text = "\n".join(
            f"- [{chunk.source}] {chunk.text}" for chunk in evidence
        ) or "NO DIRECT EVIDENCE FOUND"
        prompt = ANSWER_GENERATION_USER.substitute(
            question=question.strip(),
            question_type="unknown",
            evidence_text=evidence_text,
            job_title=job_title,
            key_requirements=", ".join(key_requirements or []) or "Not provided",
            recent_context=recent_context,
        )
        result = await self._gemini_client.generate(
            prompt=prompt,
            system_instruction=ANSWER_GENERATION_SYSTEM,
            schema=AnswerResponse,
            prompt_type="answer_generation",
        )
        if not isinstance(result, AnswerResponse):
            raise TypeError("Gemini returned an invalid answer response.")
        return result