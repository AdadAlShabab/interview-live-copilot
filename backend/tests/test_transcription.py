"""Tests for microphone capture and local transcription boundaries."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.transcription.audio_capture import AudioCapture
from app.transcription.whisper_engine import WhisperEngine


def test_audio_capture_forwards_mono_blocks():
    stream = MagicMock()
    factory = MagicMock(return_value=stream)
    received = []
    capture = AudioCapture(stream_factory=factory)

    capture.start(received.append)
    callback = factory.call_args.kwargs["callback"]
    callback(np.array([[0.1], [0.2]], dtype=np.float32), 2, None, None)
    capture.stop()

    assert received[0].shape == (2,)
    stream.start.assert_called_once()
    stream.stop.assert_called_once()
    stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_whisper_engine_transcribes_without_blocking():
    engine = WhisperEngine()
    segment = MagicMock(text=" Hello world ")
    model = MagicMock()
    model.transcribe.return_value = ([segment], None)
    engine._model = model

    result = await engine.transcribe(np.zeros(100, dtype=np.float32))

    assert result == "Hello world"
    model.transcribe.assert_called_once()