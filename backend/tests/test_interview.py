"""Tests for the local live-interview pipeline pieces."""

from app.gemini.schemas import QuestionType
from app.interview.classifier import classify
from app.interview.question_detector import is_question
from app.interview.session import InterviewState


def test_question_detector_handles_question_marks_and_starters():
    assert is_question("What did you build with Python")
    assert is_question("Tell me about a difficult project.")
    assert is_question("Hi, thanks for joining us today.")
    assert not is_question("I built an API with Python.")


def test_classifier_marks_technical_and_behavioral_questions():
    technical = classify("How did you optimize the SQL query?")
    behavioral = classify("Tell me about a conflict with a teammate.")

    assert technical.question_type == QuestionType.TECHNICAL
    assert technical.requires_technical_answer
    assert behavioral.question_type == QuestionType.BEHAVIORAL
    assert behavioral.requires_story


def test_interview_state_keeps_only_recent_context():
    state = InterviewState("session-1")
    for index in range(8):
        state.add_turn("interviewer", f"Question {index}")

    assert state.turn_count == 8
    assert "Question 0" not in state.recent_context
    assert "Question 7" in state.recent_context