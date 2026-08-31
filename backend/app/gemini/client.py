"""
Interview Copilot — Gemini API Client

Wraps google-genai with:
- Exponential backoff on 429/503
- Circuit breaker via UsageTracker
- Token estimation and recording
- Streaming support
- Structured JSON output via Pydantic
- Graceful degradation to local fallback
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional, Type, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from pydantic import BaseModel

from app.config.settings import get_settings
from app.gemini.usage_tracker import UsageTracker

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Retry configuration ─────────────────────────────────────────────────────
MAX_RETRIES = 3
BASE_BACKOFF_S = 2.0   # First retry after 2s
MAX_BACKOFF_S = 60.0   # Never wait longer than 60s


class GeminiError(Exception):
    """Base error for Gemini client failures."""


class GeminiRateLimitError(GeminiError):
    """Raised when the API returns HTTP 429."""
    def __init__(self, retry_after: float = 60.0):
        self.retry_after = retry_after
        super().__init__(f"Gemini rate limit hit. Retry after {retry_after:.0f}s.")


class GeminiAuthError(GeminiError):
    """Raised when the API key is invalid or missing."""


class GeminiUnavailableError(GeminiError):
    """Raised when Gemini returns 503 or network fails."""


class GeminiClient:
    """
    Production-grade Gemini API client.

    Usage:
        client = GeminiClient.from_settings()
        result = await client.generate(prompt="...", schema=MySchema)
    """

    def __init__(self, api_key: str, model: str, usage_tracker: UsageTracker):
        if not api_key or len(api_key) < 10:
            raise GeminiAuthError(
                "GEMINI_API_KEY is not set or is too short. "
                "Get a free key at https://aistudio.google.com/"
            )
        self._model = model
        self._tracker = usage_tracker
        # Initialize the google-genai client
        self._client = genai.Client(api_key=api_key)
        logger.info(f"[GeminiClient] Initialized with model={model}")

    @classmethod
    def from_settings(cls, usage_tracker: UsageTracker) -> "GeminiClient":
        """Factory: create from application settings."""
        settings = get_settings()
        return cls(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            usage_tracker=usage_tracker,
        )

    # ── Core generation ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        schema: Optional[Type[T]] = None,
        prompt_type: str = "generic",
        temperature: float = 0.3,
    ) -> str | T:
        """
        Generate a response from Gemini with retry/backoff.

        Args:
            prompt: The user-facing prompt content.
            system_instruction: Optional system-level context.
            schema: If provided, returns a Pydantic model instance (structured output).
            prompt_type: Label for usage tracking (e.g., "answer_generation").
            temperature: Sampling temperature. Lower = more deterministic.

        Returns:
            If schema is None: raw text string.
            If schema is provided: validated Pydantic model instance.

        Raises:
            GeminiRateLimitError: On 429 after all retries exhausted.
            GeminiAuthError: On invalid API key.
            GeminiUnavailableError: On persistent network/server failures.
        """
        # Circuit breaker check
        allowed, reason = self._tracker.can_make_request()
        if not allowed:
            raise GeminiRateLimitError(60.0) if "Rate limit" in reason else GeminiError(reason)

        last_exc: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                start_ms = time.monotonic() * 1000
                response = await self._call_api(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    schema=schema,
                    temperature=temperature,
                )
                latency_ms = time.monotonic() * 1000 - start_ms

                # Estimate tokens (metadata may be available in response)
                input_tokens = self._estimate_tokens(
                    prompt + (system_instruction or "")
                )
                output_tokens = self._estimate_tokens(
                    response.text if hasattr(response, "text") and response.text else ""
                )

                # Use actual usage metadata when available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    meta = response.usage_metadata
                    input_tokens = getattr(meta, "prompt_token_count", input_tokens) or input_tokens
                    output_tokens = getattr(meta, "candidates_token_count", output_tokens) or output_tokens

                self._tracker.record_request(
                    prompt_type=prompt_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    latency_ms=latency_ms,
                )

                logger.debug(
                    f"[GeminiClient] {prompt_type} completed in {latency_ms:.0f}ms "
                    f"({input_tokens}→{output_tokens} tokens)"
                )

                # Return parsed Pydantic model if schema provided
                if schema and response.text:
                    return schema.model_validate_json(response.text)
                return response.text or ""

            except ClientError as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                if status == 429:
                    retry_after = self._parse_retry_after(e)
                    self._tracker.record_request(
                        prompt_type=prompt_type,
                        input_tokens=0,
                        output_tokens=0,
                        success=False,
                        error_code="429",
                    )
                    self._tracker.set_rate_limited(retry_after)
                    backoff = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
                    logger.warning(
                        f"[GeminiClient] 429 Rate limit (attempt {attempt+1}/{MAX_RETRIES}). "
                        f"Backing off {backoff:.0f}s."
                    )
                    await asyncio.sleep(backoff)
                    continue

                elif status == 401:
                    self._tracker.record_request(
                        prompt_type=prompt_type, input_tokens=0, output_tokens=0,
                        success=False, error_code="401",
                    )
                    raise GeminiAuthError(
                        "Invalid or missing GEMINI_API_KEY. "
                        "Check your .env file and https://aistudio.google.com/"
                    ) from e

                else:
                    # Other 4xx — don't retry
                    self._tracker.record_request(
                        prompt_type=prompt_type, input_tokens=0, output_tokens=0,
                        success=False, error_code=str(status),
                    )
                    raise GeminiError(f"Gemini API error {status}: {e}") from e

            except APIError as e:
                last_exc = e
                # 5xx / network errors — retry with backoff
                backoff = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
                logger.warning(
                    f"[GeminiClient] API error (attempt {attempt+1}/{MAX_RETRIES}). "
                    f"Backing off {backoff:.0f}s. Error: {e}"
                )
                self._tracker.record_request(
                    prompt_type=prompt_type, input_tokens=0, output_tokens=0,
                    success=False, error_code="5xx",
                )
                await asyncio.sleep(backoff)

            except Exception as e:
                last_exc = e
                self._tracker.record_request(
                    prompt_type=prompt_type, input_tokens=0, output_tokens=0,
                    success=False, error_code="unknown",
                )
                logger.error(f"[GeminiClient] Unexpected error: {e}", exc_info=True)
                raise GeminiUnavailableError(f"Unexpected error: {e}") from e

        # All retries exhausted
        raise GeminiUnavailableError(
            f"Gemini unavailable after {MAX_RETRIES} retries. Last error: {last_exc}"
        )

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        prompt_type: str = "streaming",
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """
        Stream a Gemini response token by token.
        Yields text chunks as they arrive.
        """
        allowed, reason = self._tracker.can_make_request()
        if not allowed:
            yield f"[BLOCKED] {reason}"
            return

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
            )
            input_tokens = self._estimate_tokens(prompt + (system_instruction or ""))
            output_tokens = 0

            async for chunk in await asyncio.to_thread(
                self._stream_sync, prompt, config
            ):
                if chunk.text:
                    output_tokens += self._estimate_tokens(chunk.text)
                    yield chunk.text

            self._tracker.record_request(
                prompt_type=prompt_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

        except Exception as e:
            self._tracker.record_request(
                prompt_type=prompt_type, input_tokens=0, output_tokens=0,
                success=False, error_code="stream_error",
            )
            logger.error(f"[GeminiClient] Stream error: {e}", exc_info=True)
            yield f"[ERROR] {e}"

    async def health_check(self) -> dict:
        """
        Verify Gemini connectivity with a minimal prompt.
        Returns status dict for the /health endpoint.
        """
        settings = get_settings()

        if not settings.is_gemini_configured():
            return {
                "connected": False,
                "model": self._model,
                "error": "GEMINI_API_KEY not configured",
            }

        try:
            response = await self._call_api(
                prompt="Reply with exactly: OK",
                system_instruction="You are a health-check bot. Reply only 'OK'.",
                schema=None,
                temperature=0.0,
            )
            text = response.text if hasattr(response, "text") else str(response)
            connected = "OK" in (text or "").upper()
            return {
                "connected": connected,
                "model": self._model,
                "response_preview": (text or "")[:50],
            }
        except GeminiAuthError as e:
            return {"connected": False, "model": self._model, "error": str(e)}
        except Exception as e:
            return {"connected": False, "model": self._model, "error": str(e)}

    # ── Private ──────────────────────────────────────────────────────────────

    async def _call_api(
        self,
        prompt: str,
        system_instruction: Optional[str],
        schema: Optional[Type[BaseModel]],
        temperature: float,
    ) -> Any:
        """Run the blocking google-genai call in a thread pool."""
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema

        config = types.GenerateContentConfig(**config_kwargs)

        return await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=config,
        )

    def _stream_sync(self, prompt: str, config: Any):
        """Synchronous streaming call (run in thread via asyncio.to_thread)."""
        return self._client.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=config,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token estimation: ~4 chars per token (GPT/Gemini average).
        Used when the API doesn't return token counts.
        """
        return max(1, len(text) // 4)

    @staticmethod
    def _parse_retry_after(error: ClientError) -> float:
        """Extract retry-after seconds from a 429 error, defaulting to 60s."""
        try:
            # Some errors include retry_after in metadata
            if hasattr(error, "retry_after"):
                return float(error.retry_after)
        except (AttributeError, TypeError, ValueError):
            pass
        return 60.0
