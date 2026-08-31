"""Small sounddevice microphone capture wrapper."""

from collections.abc import Callable
from threading import Lock
from typing import Any

import numpy as np


class AudioCaptureError(RuntimeError):
    """Raised when microphone capture cannot be started or stopped."""


class AudioCapture:
    """Capture mono float32 audio without opening a device at import time."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        block_duration: float = 0.5,
        stream_factory: Callable[..., Any] | None = None,
    ):
        if sample_rate < 1 or channels < 1 or block_duration <= 0:
            raise ValueError("Invalid audio capture configuration.")
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_duration)
        self._stream_factory = stream_factory
        self._stream: Any | None = None
        self._callback: Callable[[np.ndarray], None] | None = None
        self._lock = Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self.is_recording:
            raise AudioCaptureError("Audio capture is already running.")
        self._callback = callback
        factory = self._stream_factory or self._default_stream_factory()
        try:
            self._stream = factory(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.block_size,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioCaptureError(f"Unable to start microphone capture: {exc}") from exc

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            raise AudioCaptureError(f"Unable to stop microphone capture: {exc}") from exc
        finally:
            self._callback = None

    def _on_audio(self, data: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            return
        callback = self._callback
        if callback is not None:
            audio = np.asarray(data, dtype=np.float32).copy()
            callback(audio[:, 0] if self.channels == 1 else audio)

    @staticmethod
    def _default_stream_factory():
        import sounddevice as sd

        return sd.InputStream