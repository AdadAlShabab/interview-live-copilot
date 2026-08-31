"""
Interview Copilot — Pydantic Output Schemas

All schemas that Gemini must return as structured JSON.
Using Pydantic ensures validation and prevents hallucinations
from causing silent data corruption.

These are the single source of truth for Gemini's output contracts.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    RESUME = "resume"
    COMPANY = "company"
    SALARY = "salary"
    MOTIVATION = "motivation"
    STRENGTHS = "strengths"
    WEAKNESSES = "weaknesses"
    SITUATIONAL = "situational"
    FOLLOWUP = "followup"
    OTHER = "other"


class MatchStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    PARTIAL = "partial"
    NONE = "none"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Resume Extraction ────────────────────────────────────────────────────────

class EducationEntry(BaseModel):
    institution: str = Field(description="Name of university or institution")
    degree: str = Field(description="Degree title e.g. BSc Computer Science")
    field_of_study: Optional[str] = Field(default=None)
    graduation_year: Optional[str] = Field(default=None)
    gpa: Optional[str] = Field(default=None)


class ExperienceEntry(BaseModel):
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title")
    start_date: Optional[str] = Field(default=None, description="e.g. Jan 2020")
    end_date: Optional[str] = Field(default=None, description="e.g. Present")
    location: Optional[str] = Field(default=None)
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Concrete responsibilities — not generic descriptions",
    )
    technologies: list[str] = Field(
        default_factory=list,
        description="Technologies, tools, languages used in this role",
    )
    achievements: list[str] = Field(
        default_factory=list,
        description="Quantifiable achievements and business impact",
    )
    projects: list[str] = Field(
        default_factory=list,
        description="Named projects the candidate worked on",
    )


class BehavioralStory(BaseModel):
    title: str = Field(description="Short label, e.g. 'Automation win'")
    theme: str = Field(
        description="Core theme: leadership|conflict|failure|success|innovation|pressure"
    )
    situation: str = Field(description="STAR: Situation")
    task: str = Field(description="STAR: Task")
    action: str = Field(description="STAR: Action taken by the candidate")
    result: str = Field(description="STAR: Quantifiable result or outcome")
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that trigger this story retrieval",
    )


class CandidateProfile(BaseModel):
    """
    Full structured representation of a resume.
    Returned by Gemini after resume extraction.
    IMPORTANT: Gemini must NOT invent any field. If data is missing, omit it.
    """
    name: Optional[str] = Field(default=None)
    current_role: Optional[str] = Field(default=None)
    years_experience: Optional[float] = Field(default=None)
    location: Optional[str] = Field(default=None)
    industries: list[str] = Field(default_factory=list)
    summary: Optional[str] = Field(
        default=None,
        description="2–3 sentence professional summary",
    )
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and domain skills",
    )
    behavioral_stories: list[BehavioralStory] = Field(
        default_factory=list,
        description="Pre-extracted STAR stories for behavioral questions",
    )
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


# ── Job Description Analysis ─────────────────────────────────────────────────

class SkillMatch(BaseModel):
    skill: str = Field(description="Skill or requirement from the JD")
    strength: MatchStrength = Field(description="How well the candidate matches")
    evidence: Optional[str] = Field(
        default=None,
        description="Candidate evidence supporting this match, or None if no match",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Bridging notes for partial matches",
    )


class JobAnalysis(BaseModel):
    """Structured result of job description analysis."""
    job_title: str
    company: Optional[str] = None
    domain: Optional[str] = Field(
        default=None,
        description="Business domain e.g. fintech, healthcare, e-commerce",
    )
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    technical_requirements: list[str] = Field(default_factory=list)
    behavioral_competencies: list[str] = Field(default_factory=list)
    likely_interview_questions: list[str] = Field(
        default_factory=list,
        description="Top 10 most likely interview questions for this role",
    )
    important_keywords: list[str] = Field(default_factory=list)
    skill_matches: list[SkillMatch] = Field(
        default_factory=list,
        description="Candidate ↔ job requirement match analysis",
    )


# ── Answer Generation ────────────────────────────────────────────────────────

class AnswerResponse(BaseModel):
    """
    Gemini's structured answer to a detected interview question.
    Anti-hallucination: Gemini is explicitly instructed that
    direct_experience=False means no fabrication allowed.
    """
    question_type: QuestionType
    confidence: ConfidenceLevel
    direct_experience: bool = Field(
        description=(
            "True ONLY when the candidate's actual resume contains "
            "direct evidence for this question. "
            "False means no matching experience was found."
        )
    )
    suggested_answer: str = Field(
        description=(
            "The full suggested spoken answer. "
            "MUST be grounded ONLY in the provided evidence. "
            "If direct_experience is False, must start with a clear acknowledgment."
        )
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="3–5 bullet points summarizing the answer",
    )
    evidence_used: list[str] = Field(
        default_factory=list,
        description="Exact evidence items from candidate profile used to form the answer",
    )
    transferable_approach: Optional[str] = Field(
        default=None,
        description=(
            "When direct_experience=False, suggest a transferable-skills bridge. "
            "Otherwise None."
        ),
    )
    likely_followups: list[str] = Field(
        default_factory=list,
        description="2–3 likely follow-up questions the interviewer may ask",
    )
    sql_example: Optional[str] = Field(
        default=None,
        description="For SQL technical questions: a relevant SQL code example",
    )
    python_example: Optional[str] = Field(
        default=None,
        description="For Python technical questions: a relevant code snippet",
    )


# ── Question Classification ───────────────────────────────────────────────────

class QuestionClassification(BaseModel):
    """Local + Gemini question classification result."""
    question_text: str
    question_type: QuestionType
    topic: Optional[str] = Field(
        default=None,
        description="Specific topic e.g. 'SQL optimization', 'leadership'",
    )
    confidence: ConfidenceLevel
    requires_technical_answer: bool = False
    requires_story: bool = False


# ── Conversation Summary ──────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    """Rolling compressed summary of the interview so far."""
    turn_count: int
    topics_discussed: list[str] = Field(default_factory=list)
    claims_made: list[str] = Field(
        default_factory=list,
        description="Key claims the candidate has made (to avoid contradictions)",
    )
    interviewer_interests: list[str] = Field(
        default_factory=list,
        description="Topics the interviewer has shown repeated interest in",
    )
    summary_text: str = Field(
        description="2–4 sentence plain-English summary of the conversation so far"
    )
