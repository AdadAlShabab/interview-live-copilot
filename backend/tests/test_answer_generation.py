"""Tests for evidence-grounded answer generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.gemini.client import GeminiClient
from app.gemini.schemas import AnswerResponse, ConfidenceLevel, QuestionType
from app.interview.answer_generator import AnswerGenerator
from app.storage.models import Candidate, Experience


@pytest.mark.asyncio
async def test_answer_generator_sends_retrieved_evidence(tmp_path):
    candidate = Candidate(id=7, name="Jane Doe", skills=["Python"], experiences=[
        Experience(company="Acme", role="Engineer", responsibilities=["Built APIs"])
    ], stories=[])
    answer = AnswerResponse(
        question_type=QuestionType.TECHNICAL,
        confidence=ConfidenceLevel.HIGH,
        direct_experience=True,
        suggested_answer="I built APIs at Acme.",
    )
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock(return_value=answer)

    result = await AnswerGenerator(client, tmp_path).generate(
        "How did you build APIs?", candidate, job_title="Backend Engineer"
    )

    assert result == answer
    prompt = client.generate.call_args.kwargs["prompt"]
    assert "Built APIs" in prompt
    assert "Acme" in prompt
    assert client.generate.call_args.kwargs["prompt_type"] == "answer_generation"