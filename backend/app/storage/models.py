"""
Interview Copilot — ORM Models

SQLAlchemy models for persisting candidate profiles and interview state.
Designed to mirror the Pydantic schemas.
"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import relationship

from app.storage.database import Base


class JSONEncodedList(TypeDecorator):
    """Custom SQLAlchemy type for storing Python lists as JSON strings in SQLite."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: str | None, dialect: Any) -> list[Any] | None:
        if value is None:
            return None
        return json.loads(value)


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String, nullable=True)
    current_role = Column(String, nullable=True)
    years_experience = Column(Float, nullable=True)
    location = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    industries = Column(JSONEncodedList, default="[]")
    skills = Column(JSONEncodedList, default="[]")
    certifications = Column(JSONEncodedList, default="[]")
    languages = Column(JSONEncodedList, default="[]")

    # Relationships
    experiences = relationship(
        "Experience", back_populates="candidate", cascade="all, delete-orphan"
    )
    education = relationship(
        "Education", back_populates="candidate", cascade="all, delete-orphan"
    )
    stories = relationship(
        "BehavioralStory", back_populates="candidate", cascade="all, delete-orphan"
    )


class Experience(Base):
    __tablename__ = "experiences"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    company = Column(String, nullable=False)
    role = Column(String, nullable=False)
    start_date = Column(String, nullable=True)
    end_date = Column(String, nullable=True)
    location = Column(String, nullable=True)

    responsibilities = Column(JSONEncodedList, default="[]")
    technologies = Column(JSONEncodedList, default="[]")
    achievements = Column(JSONEncodedList, default="[]")
    projects = Column(JSONEncodedList, default="[]")

    candidate = relationship("Candidate", back_populates="experiences")


class Education(Base):
    __tablename__ = "education"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    institution = Column(String, nullable=False)
    degree = Column(String, nullable=False)
    field_of_study = Column(String, nullable=True)
    graduation_year = Column(String, nullable=True)
    gpa = Column(String, nullable=True)

    candidate = relationship("Candidate", back_populates="education")


class BehavioralStory(Base):
    __tablename__ = "behavioral_stories"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    title = Column(String, nullable=False)
    theme = Column(String, nullable=False)
    situation = Column(Text, nullable=False)
    task = Column(Text, nullable=False)
    action = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    keywords = Column(JSONEncodedList, default="[]")

    candidate = relationship("Candidate", back_populates="stories")


class JobDescription(Base):
    """Stores analyzed job descriptions."""
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    title = Column(String, nullable=False)
    company = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)

    required_skills = Column(JSONEncodedList, default="[]")
    preferred_skills = Column(JSONEncodedList, default="[]")
    responsibilities = Column(JSONEncodedList, default="[]")
    technical_requirements = Column(JSONEncodedList, default="[]")
    behavioral_competencies = Column(JSONEncodedList, default="[]")
    likely_questions = Column(JSONEncodedList, default="[]")
    important_keywords = Column(JSONEncodedList, default="[]")

    # Match analysis stored as JSON
    skill_matches = Column(JSONEncodedList, default="[]")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=True)

    # Rolling memory state stored as JSON
    memory_state = Column(JSONEncodedList, default="{}")

    # Relationships
    turns = relationship(
        "InterviewTurn", back_populates="session", cascade="all, delete-orphan"
    )


class InterviewTurn(Base):
    """A single Q&A turn in an interview."""
    __tablename__ = "interview_turns"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Input
    speaker = Column(String, nullable=False)  # "interviewer" or "candidate"
    text = Column(Text, nullable=False)

    # If this turn was a detected question, store the AI response here
    is_question = Column(Integer, default=0)  # Boolean representation
    question_type = Column(String, nullable=True)
    ai_suggested_answer = Column(Text, nullable=True)
    ai_evidence_used = Column(JSONEncodedList, default="[]")
    ai_latency_ms = Column(Float, nullable=True)

    session = relationship("InterviewSession", back_populates="turns")
