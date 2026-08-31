"""Local question classification with no Gemini request."""

from app.gemini.schemas import ConfidenceLevel, QuestionClassification, QuestionType


def classify(question: str) -> QuestionClassification:
    """Classify common interview question patterns using inexpensive heuristics."""
    text = question.lower()
    if any(term in text for term in ("sql", "python", "api", "database", "code")):
        question_type = QuestionType.TECHNICAL
        requires_technical_answer = True
        requires_story = False
    elif any(term in text for term in ("tell me about", "conflict", "challenge", "failure")):
        question_type = QuestionType.BEHAVIORAL
        requires_technical_answer = False
        requires_story = True
    elif any(term in text for term in ("why do you want", "why this role", "motivat")):
        question_type = QuestionType.MOTIVATION
        requires_technical_answer = False
        requires_story = False
    else:
        question_type = QuestionType.OTHER
        requires_technical_answer = False
        requires_story = False
    return QuestionClassification(
        question_text=question,
        question_type=question_type,
        topic=None,
        confidence=ConfidenceLevel.MEDIUM,
        requires_technical_answer=requires_technical_answer,
        requires_story=requires_story,
    )