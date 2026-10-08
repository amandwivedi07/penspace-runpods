"""Assemble chunk audio into a mastered, streamable summary track."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .chunker import Chunk
from .config import Config
from .synth import ChunkAudio

log = logging.getLogger(__name__)

# Silence trimmed from chunk edges before joining, so our own gap timings are
# the only thing controlling pace.
TRIM_TOP_DB = 35


@dataclass
class ChunkTiming:
    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


def _silence(ms: int, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * ms / 1000), dtype=np.float32)


def _trim(wav: np.ndarray) -> np.ndarray:
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    return trimmed if trimmed.size else wav


def _stretch(wav: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    """Change tempo by [rate] without changing pitch (ffmpeg atempo, WSOLA).

    Runs per sentence, on the trimmed speech only, so silences inserted
    afterwards are never stretched.
    """
    if rate == 1.0 or wav.size == 0:
        return wav
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-filter:a", f"atempo={rate}",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
        ],
        input=np.ascontiguousarray(wav, dtype=np.float32).tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def assemble(
    audios: List[ChunkAudio],
    chunks: List[Chunk],
    cfg: Config,
) -> tuple[np.ndarray, int, List[ChunkTiming]]:
    """Join chunks with paragraph-aware pauses, returning audio and a timing map."""
    by_index = {a.index: a for a in audios}
    sample_rate = audios[0].sample_rate

    pieces: List[np.ndarray] = [_silence(cfg.lead_in_ms, sample_rate)]
    timings: List[ChunkTiming] = []
    cursor = cfg.lead_in_ms / 1000.0

    for i, chunk in enumerate(chunks):
        audio = by_index.get(chunk.index)
        if audio is None:
            raise KeyError(f"missing audio for chunk {chunk.index}")

        wav = _stretch(_trim(audio.wav), sample_rate, cfg.speech_rate)
        duration = len(wav) / sample_rate
        pieces.append(wav)
        timings.append(ChunkTiming(chunk.index, cursor, cursor + duration, chunk.text))
        cursor += duration

        if i < len(chunks) - 1:
            # This gap sits BETWEEN chunk i and chunk i+1, so both sides have
            # a claim on it. A heading wants room after it, and — the part a
            # listener actually notices — the chunk before a heading wants room
            # so the previous thought lands before the next one is named.
            # Whichever side asks for more wins.
            if chunk.is_heading:
                gap = cfg.heading_gap_ms
            elif chunk.ends_paragraph:
                gap = cfg.paragraph_gap_ms
            else:
                gap = cfg.sentence_gap_ms
            if chunks[i + 1].is_heading:
                gap = max(gap, cfg.heading_lead_gap_ms)
            pieces.append(_silence(gap, sample_rate))
            cursor += gap / 1000.0

    pieces.append(_silence(cfg.tail_ms, sample_rate))
    return np.concatenate(pieces), sample_rate, timings


def _limit_peaks(
    wav: np.ndarray,
    ceiling: float,
    sample_rate: int,
    lookahead_ms: float = 2.0,
    smooth_ms: float = 20.0,
) -> np.ndarray:
    """Shave transient peaks, leaving the body of the signal alone.

    The naive alternative -- scaling the whole track by ceiling/peak -- drops
    integrated loudness by however much the single loudest transient
    overshoots. That is how a -16 LUFS target silently produced -20 LUFS audio.
    Here the gain reduction is applied only around the offending samples, so
    loudness stays on target.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    magnitude = np.abs(wav)
    if magnitude.size == 0 or float(magnitude.max()) <= ceiling:
        return wav

    gain = np.ones_like(wav, dtype=np.float32)
    over = magnitude > ceiling
    gain[over] = ceiling / magnitude[over]

    # Lookahead: pull the gain down slightly *before* each peak arrives.
    lookahead = max(1, int(sample_rate * lookahead_ms / 1000))
    gain = minimum_filter1d(gain, size=2 * lookahead + 1, mode="nearest")

    # Smooth attack/release so the reduction is not audible as a click.
    smooth = max(1, int(sample_rate * smooth_ms / 1000))
    gain = uniform_filter1d(gain, size=2 * smooth + 1, mode="nearest")

    return np.clip(wav * gain, -ceiling, ceiling)


def master(wav: np.ndarray, sample_rate: int, cfg: Config) -> np.ndarray:
    """Loudness-normalize to the spoken-word target and limit true peaks."""
    import warnings

    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(wav)

    if np.isfinite(loudness):
        with warnings.catch_warnings():
            # pyloudnorm warns about overshoot that the limiter below handles.
            warnings.simplefilter("ignore")
            wav = pyln.normalize.loudness(wav, loudness, cfg.target_lufs)
    else:
        log.warning("could not measure loudness; skipping normalization")

    ceiling = 10 ** (cfg.peak_ceiling_db / 20.0)
    wav = _limit_peaks(wav, ceiling, sample_rate).astype(np.float32)

    achieved = meter.integrated_loudness(wav)
    peak_db = 20 * np.log10(max(float(np.max(np.abs(wav))), 1e-9))
    log.info(
        "mastered to %.2f LUFS (target %.1f), peak %.2f dBFS",
        achieved,
        cfg.target_lufs,
        peak_db,
    )
    return wav


def encode(
    wav: np.ndarray,
    sample_rate: int,
    out_path: Path,
    cfg: Config,
    title: Optional[str] = None,
    artist: Optional[str] = None,
) -> Path:
    """Write an AAC/m4a file with the moov atom up front for instant streaming."""
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode m4a; install it on the VM")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".tmp.wav")
    sf.write(wav_path, wav, sample_rate)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-c:a", "aac", "-b:a", cfg.bitrate, "-ac", "1",
        "-movflags", "+faststart",
    ]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if artist:
        cmd += ["-metadata", f"artist={artist}"]
    cmd.append(str(out_path))

    try:
        subprocess.run(cmd, check=True)
    finally:
        wav_path.unlink(missing_ok=True)

    return out_path


def write_timing_map(
    timings: List[ChunkTiming],
    out_path: Path,
    duration: float,
    metadata: dict,
) -> Path:
    """Persist chunk timings alongside the audio.

    Costs nothing now and is what lets the app do sentence highlighting and
    scrub-to-paragraph later without re-rendering the catalog.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "duration": round(duration, 3),
        "chunks": [t.to_dict() for t in timings],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path
