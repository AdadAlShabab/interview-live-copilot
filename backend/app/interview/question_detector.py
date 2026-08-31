"""Zero-cost heuristics for deciding whether transcript text is a question."""

import re


QUESTION_STARTS = (
    "are ", "can ", "could ", "did ", "do ", "have ", "how ",
    "what ", "when ", "where ", "which ", "why ", "will ", "would ",
    "tell me ", "describe ", "walk me ",
)

GREETING_STARTS = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "nice to meet you",
    "thanks for joining",
    "thank you for joining",
)


def is_question(text: str) -> bool:
    """Return true for likely interviewer questions in a transcript chunk."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False

    if normalized.endswith(("?", "？")):
        return True

    if normalized.startswith(QUESTION_STARTS):
        return True

    normalized_no_punct = normalized.strip(" ,;:!.-")
    if normalized_no_punct in GREETING_STARTS:
        return True

    return any(normalized_no_punct.startswith(f"{greeting}") for greeting in GREETING_STARTS)
