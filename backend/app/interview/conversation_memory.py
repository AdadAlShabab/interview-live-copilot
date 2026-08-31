"""Rolling conversation memory and local follow-up detection."""

import re

from app.gemini.client import GeminiClient
from app.gemini.prompts import CONVERSATION_SUMMARY_SYSTEM, CONVERSATION_SUMMARY_USER
from app.gemini.schemas import ConversationSummary


class ConversationMemory:
    """Keep turns bounded and periodically compress them with Gemini."""

    def __init__(self, gemini_client: GeminiClient | None = None, summary_interval: int = 10):
        self.gemini_client = gemini_client
        self.summary_interval = summary_interval
        self.turns: list[str] = []
        self.summary: ConversationSummary | None = None
        self.total_turns = 0
        self.last_summary_turn = 0

    def add_turn(self, speaker: str, text: str) -> None:
        self.total_turns += 1
        self.turns.append(f"{speaker}: {text}")

    def should_summarize(self) -> bool:
        return bool(
            self.gemini_client
            and self.total_turns - self.last_summary_turn >= self.summary_interval
        )

    async def summarize(self) -> ConversationSummary | None:
        if not self.should_summarize():
            return self.summary
        start = self.last_summary_turn + 1
        end = self.total_turns
        prompt = CONVERSATION_SUMMARY_USER.substitute(
            start_turn=start,
            end_turn=end,
            turns_text="\n".join(self.turns[-self.summary_interval:]),
        )
        result = await self.gemini_client.generate(
            prompt=prompt,
            system_instruction=CONVERSATION_SUMMARY_SYSTEM,
            schema=ConversationSummary,
            prompt_type="conversation_summary",
        )
        if not isinstance(result, ConversationSummary):
            raise TypeError("Gemini returned an invalid conversation summary.")
        self.summary = result
        self.last_summary_turn = self.total_turns
        self.turns = self.turns[-6:]
        return result

    @property
    def context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Summary: {self.summary.summary_text}")
        parts.append("Recent turns:\n" + "\n".join(self.turns[-6:]))
        return "\n".join(parts)


def is_followup(question: str) -> bool:
    """Detect questions that refer to the immediately preceding answer."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    return bool(re.search(r"\b(and|also|what about|how about|why so|can you elaborate)\b", text))