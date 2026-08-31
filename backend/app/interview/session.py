"""In-memory state for one live interview connection."""

from dataclasses import dataclass, field


@dataclass
class InterviewState:
    """Keep the recent transcript context bounded for Gemini prompts."""

    session_uuid: str
    turn_count: int = 0
    recent_turns: list[str] = field(default_factory=list)

    def add_turn(self, speaker: str, text: str) -> None:
        self.turn_count += 1
        self.recent_turns.append(f"{speaker}: {text}")
        self.recent_turns = self.recent_turns[-6:]

    @property
    def recent_context(self) -> str:
        return "\n".join(self.recent_turns) or "None"