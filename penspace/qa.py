"""ASR round-trip quality gate.

Neural TTS occasionally repeats, skips or truncates spans of text. In a reading
app a single garbled sentence is glaringly obvious, so every chunk is
transcribed back with Whisper and compared to its source text. Chunks over the
WER threshold are regenerated with a different seed.

Both sides go through the same normalization before comparison: Whisper emits
digits and abbreviations ("2019", "Mr.") while our source text is already
spelled out ("twenty nineteen", "Mister"), and comparing them raw would flag
every number as an error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import runtime
from .config import Config
from .normalize import normalize

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
_PUNCT = re.compile(r"[^\w\s]")


@dataclass
class QAResult:
    index: int
    wer: float
    transcript: str
    passed: bool


def _canonical(text: str) -> str:
    text = normalize(text)
    text = _PUNCT.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class QAGate:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._wer = None

    def load(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        from jiwer import wer as jiwer_wer

        self._wer = jiwer_wer
        device, compute_type = runtime.whisper_device(
            runtime.resolve_device(self.cfg.device)
        )
        log.info("Loading Whisper %s on %s", self.cfg.whisper_model, device)
        self._model = WhisperModel(
            self.cfg.whisper_model, device=device, compute_type=compute_type
        )

    def transcribe(self, wav: np.ndarray, sample_rate: int) -> str:
        self.load()
        import librosa

        if sample_rate != WHISPER_SAMPLE_RATE:
            wav = librosa.resample(
                wav, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE
            )
        segments, _ = self._model.transcribe(
            wav.astype(np.float32), language="en", beam_size=1
        )
        return " ".join(seg.text for seg in segments).strip()

    def transcribe_file(self, path: str) -> str:
        """Transcribe a reference recording, to use as voice-clone ref_text."""
        import librosa

        wav, sr = librosa.load(path, sr=WHISPER_SAMPLE_RATE, mono=True)
        return self.transcribe(wav, sr)

    def check(
        self, index: int, wav: np.ndarray, sample_rate: int, expected: str
    ) -> QAResult:
        transcript = self.transcribe(wav, sample_rate)
        reference = _canonical(expected)
        hypothesis = _canonical(transcript)

        if not reference:
            return QAResult(index, 0.0, transcript, True)
        if not hypothesis:
            # Silence or pure noise: worst possible outcome, always retry.
            return QAResult(index, 1.0, transcript, False)

        score = float(self._wer(reference, hypothesis))
        passed = score <= self.cfg.max_wer
        if not passed:
            log.warning(
                "chunk %d failed QA (WER %.3f > %.3f)\n  expected: %s\n  heard:    %s",
                index,
                score,
                self.cfg.max_wer,
                reference[:120],
                hypothesis[:120],
            )
        return QAResult(index, score, transcript, passed)
