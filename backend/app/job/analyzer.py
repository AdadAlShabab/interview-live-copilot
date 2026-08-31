"""Gemini-backed job description analysis."""

from app.gemini.client import GeminiClient
from app.gemini.prompts import JOB_ANALYSIS_SYSTEM, JOB_ANALYSIS_USER
from app.gemini.schemas import JobAnalysis
from app.storage.models import Candidate


class JobAnalyzer:
    """Build the compact candidate context and request structured analysis."""

    def __init__(self, gemini_client: GeminiClient):
        self._gemini_client = gemini_client

    async def analyze(
        self,
        job_description: str,
        candidate: Candidate | None = None,
    ) -> JobAnalysis:
        values = self._candidate_summary(candidate)
        prompt = JOB_ANALYSIS_USER.substitute(
            job_description=job_description.strip(),
            **values,
        )
        result = await self._gemini_client.generate(
            prompt=prompt,
            system_instruction=JOB_ANALYSIS_SYSTEM,
            schema=JobAnalysis,
            prompt_type="job_analysis",
        )
        if not isinstance(result, JobAnalysis):
            raise TypeError("Gemini returned an invalid job analysis.")
        return result

    @staticmethod
    def _candidate_summary(candidate: Candidate | None) -> dict[str, str]:
        if candidate is None:
            return {
                "candidate_name": "Not provided",
                "current_role": "Not provided",
                "years_experience": "Not provided",
                "skills_list": "Not provided",
                "companies_list": "Not provided",
            }

        companies = [experience.company for experience in candidate.experiences]
        return {
            "candidate_name": candidate.name or "Not provided",
            "current_role": candidate.current_role or "Not provided",
            "years_experience": str(candidate.years_experience or "Not provided"),
            "skills_list": ", ".join(candidate.skills or []) or "Not provided",
            "companies_list": ", ".join(dict.fromkeys(companies)) or "Not provided",
        }