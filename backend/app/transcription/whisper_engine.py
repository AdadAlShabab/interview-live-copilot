"""Lazy faster-whisper transcription service."""

import asyncio
from typing import Any


class TranscriptionError(RuntimeError):
    """Raised when local speech-to-text fails."""


class WhisperEngine:
    """Run faster-whisper off the event loop and load its model on demand."""

    def __init__(self, model_size: str = "small", device: str = "cpu", threads: int = 4):
        self.model_size = model_size
        self.device = device
        self.threads = threads
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16",
                cpu_threads=self.threads,
            )
        except Exception as exc:
            raise TranscriptionError(f"Unable to load Whisper model: {exc}") from exc

    async def transcribe(self, audio: Any, language: str | None = None) -> str:
        """Transcribe a numpy audio buffer without blocking the async loop."""
        self.load()
        try:
            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio,
                language=language,
                vad_filter=True,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc