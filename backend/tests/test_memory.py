"""Tests for rolling conversation memory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.gemini.client import GeminiClient
from app.gemini.schemas import ConversationSummary
from app.interview.conversation_memory import ConversationMemory, is_followup


def test_followup_detection():
    assert is_followup("And what about monitoring?")
    assert is_followup("Can you elaborate?")
    assert not is_followup("Why did you choose Python?")


@pytest.mark.asyncio
async def test_memory_summarizes_at_interval():
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock(return_value=ConversationSummary(
        turn_count=10,
        summary_text="Discussed APIs and testing.",
    ))
    memory = ConversationMemory(client, summary_interval=3)
    for index in range(3):
        memory.add_turn("interviewer", f"Question {index}")

    summary = await memory.summarize()

    assert summary.turn_count == 10
    assert "Question 0" in client.generate.call_args.kwargs["prompt"]
    assert len(memory.turns) == 3
    assert not memory.should_summarize()