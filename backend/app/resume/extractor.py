"""
Interview Copilot — Resume Extractor

Handles raw text extraction from PDF, DOCX, TXT, and MD files.
Uses Gemini to parse the unstructured text into a CandidateProfile.
"""

import io
import logging
import re
from pathlib import Path

import docx
import markdown
import pdfplumber

from app.gemini.client import GeminiClient
from app.gemini.prompts import RESUME_EXTRACTION_SYSTEM, RESUME_EXTRACTION_USER
from app.gemini.schemas import CandidateProfile

logger = logging.getLogger(__name__)


class ResumeExtractionError(Exception):
    """Raised when resume extraction fails."""


class ResumeExtractor:
    """Extracts and parses resumes into structured profiles."""

    def __init__(self, gemini_client: GeminiClient):
        self._client = gemini_client

    async def process_file(self, file_path: Path, mime_type: str = "") -> CandidateProfile:
        """
        End-to-end pipeline:
        1. Extract raw text from file
        2. Clean and normalize text
        3. Send to Gemini for structured extraction
        """
        # 1. Extract text
        raw_text = self._extract_text(file_path, mime_type)
        if not raw_text.strip():
            raise ResumeExtractionError("Extracted text is empty. Unreadable file?")

        # 2. Clean text (remove excessive newlines/spaces to save tokens)
        clean_text = self._normalize_text(raw_text)

        # 3. Call Gemini
        logger.info(f"Sending resume to Gemini ({len(clean_text)} chars)...")
        prompt = RESUME_EXTRACTION_USER.substitute(resume_text=clean_text)

        try:
            profile: CandidateProfile = await self._client.generate(
                prompt=prompt,
                system_instruction=RESUME_EXTRACTION_SYSTEM,
                schema=CandidateProfile,
                prompt_type="resume_extraction",
            )
            return profile
        except Exception as e:
            logger.error(f"Gemini resume extraction failed: {e}")
            raise ResumeExtractionError(f"Failed to parse resume with AI: {e}") from e

    def _extract_text(self, file_path: Path, mime_type: str) -> str:
        """Route to appropriate extractor based on extension or mime_type."""
        ext = file_path.suffix.lower()

        if ext == ".pdf" or "pdf" in mime_type:
            return self._extract_pdf(file_path)
        elif ext in [".docx", ".doc"] or "word" in mime_type:
            return self._extract_docx(file_path)
        elif ext in [".md", ".markdown"] or "markdown" in mime_type:
            return self._extract_md(file_path)
        elif ext == ".txt" or "text" in mime_type:
            return file_path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ResumeExtractionError(f"Unsupported file format: {ext}")

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using pdfplumber (good at tables/columns)."""
        text_parts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # extract_text handles columns better than PyMuPDF by default
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception as e:
            raise ResumeExtractionError(f"PDF extraction failed: {e}") from e

    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise ResumeExtractionError(f"DOCX extraction failed: {e}") from e

    def _extract_md(self, file_path: Path) -> str:
        """Extract text from Markdown (strip HTML)."""
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            # For extraction, we just need the raw text, markdown is fine for LLM.
            # We don't necessarily need to render it to HTML and strip it.
            return text
        except Exception as e:
            raise ResumeExtractionError(f"Markdown extraction failed: {e}") from e

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove excessive whitespace to save tokens."""
        # Replace 3 or more newlines with 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()
