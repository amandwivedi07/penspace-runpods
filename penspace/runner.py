"""Batch render orchestrator: text -> chunks -> audio -> QA -> master -> S3.

Resumable at chunk granularity. Re-running the same manifest re-uploads nothing
and re-synthesizes only chunks that are missing from the local cache, which
matters because you are paying per second for the GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import audio as audio_mod
from .chunker import Chunk, chunk_text
from .config import Config
from .normalize import normalize
from .qa import QAGate
from .storage import S3Storage, render_key
from .synth import ChunkAudio, Synthesizer

log = logging.getLogger(__name__)

RETRY_SEED_BASE = 1_000


@dataclass
class SummaryJob:
    id: str
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    # The language to narrate IN — "German", "Japanese", as the model names
    # them. Absent means the endpoint's configured default, which is what every
    # job was before this and keeps older callers working unchanged.
    language: Optional[str] = None


@dataclass
class RenderResult:
    id: str
    render_id: str
    duration: float
    chunk_count: int
    audio_path: Optional[Path] = None
    timing_path: Optional[Path] = None
    audio_key: Optional[str] = None
    timing_key: Optional[str] = None
    failed_chunks: List[int] = field(default_factory=list)
    max_wer: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "render_id": self.render_id,
            "duration": round(self.duration, 3),
            "chunk_count": self.chunk_count,
            "audio_key": self.audio_key,
            "timing_key": self.timing_key,
            "failed_chunks": self.failed_chunks,
            "max_wer": round(self.max_wer, 4),
        }


MIN_REF_SECONDS = 3.0
IDEAL_REF_SECONDS = 10.0


def _reference_fingerprint(cfg: Config) -> str:
    """Hash the reference recording's bytes, not just its path."""
    if not cfg.is_clone or not cfg.ref_audio:
        return ""
    path = Path(cfg.ref_audio)
    if not path.exists():
        return cfg.ref_audio  # URL or base64: fall back to the string itself
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{digest}:{cfg.ref_text or ''}:{cfg.x_vector_only}"


def prepare_clone(cfg: Config) -> Config:
    """Validate the reference recording and fill in its transcript.

    ICL cloning needs the transcript of the reference audio. Rather than making
    you type it, transcribe it with the Whisper model already present for the
    QA gate.
    """
    if not cfg.is_clone:
        return cfg
    if not cfg.ref_audio:
        raise ValueError("voice_mode='clone' requires --ref-audio")

    path = Path(cfg.ref_audio)
    if path.exists():
        import soundfile as sf

        info = sf.info(str(path))
        log.info(
            "reference: %.1fs, %d Hz, %d channel(s)",
            info.duration,
            info.samplerate,
            info.channels,
        )
        if info.duration < MIN_REF_SECONDS:
            raise ValueError(
                f"reference audio is {info.duration:.1f}s; the model needs at "
                f"least {MIN_REF_SECONDS:.0f}s"
            )
        if info.duration < IDEAL_REF_SECONDS:
            log.warning(
                "reference is only %.1fs; %.0f-30s of clean speech clones "
                "noticeably better",
                info.duration,
                IDEAL_REF_SECONDS,
            )

    if cfg.ref_text or cfg.x_vector_only:
        return cfg

    log.info("no --ref-text given; transcribing the reference with Whisper")
    cfg.ref_text = QAGate(cfg).transcribe_file(str(path))
    log.info("reference transcript: %s", cfg.ref_text)
    if not cfg.ref_text.strip():
        raise ValueError(
            "Whisper heard nothing in the reference audio. Check it contains "
            "clear speech, or pass --ref-text manually."
        )
    return cfg


def load_manifest(path: Path) -> List[SummaryJob]:
    """Read a JSONL manifest of summaries.

    Each line: {"id", "title"?, "author"?, and one of "text" | "text_file"}.
    """
    jobs: List[SummaryJob] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if "id" not in row:
            raise ValueError(f"{path}:{lineno} missing required field 'id'")

        text = row.get("text")
        if text is None:
            text_file = row.get("text_file")
            if not text_file:
                raise ValueError(f"{path}:{lineno} needs either 'text' or 'text_file'")
            text = (path.parent / text_file).read_text()

        jobs.append(
            SummaryJob(
                id=row["id"],
                text=text,
                title=row.get("title"),
                author=row.get("author"),
            )
        )
    return jobs


class Runner:
    def __init__(
        self,
        cfg: Config,
        work_dir: Path,
        storage: Optional[S3Storage] = None,
    ):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.storage = storage
        self.synth = Synthesizer(cfg)
        self.qa = QAGate(cfg) if cfg.qa_enabled else None

    def _render_id(self, normalized_text: str, language: str) -> str:
        """Identity of this render: change the text or the voice, get a new id."""
        fingerprint = "\x00".join(
            [
                normalized_text,
                self.cfg.active_model_path,
                self.cfg.voice_mode,
                # Swapping the reference recording changes the voice, so it has
                # to change the render id too.
                _reference_fingerprint(self.cfg),
                self.cfg.speaker,
                # THE JOB'S language, not the endpoint's. The same text narrated
                # in two languages is two different recordings, and an id that
                # could not tell them apart would let the S3 skip hand back the
                # wrong one.
                language,
                self.cfg.instruct,
                str(self.cfg.max_chunk_chars),
                str(self.cfg.target_lufs),
                self.cfg.bitrate,
            ]
        )
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

    # --- chunk-level cache -------------------------------------------------

    def _chunk_path(self, out_dir: Path, index: int) -> Path:
        return out_dir / "chunks" / f"{index:04d}.wav"

    def _load_cached(self, out_dir: Path, chunks: List[Chunk]) -> Dict[int, ChunkAudio]:
        import soundfile as sf

        cached: Dict[int, ChunkAudio] = {}
        for chunk in chunks:
            path = self._chunk_path(out_dir, chunk.index)
            if not path.exists():
                continue
            wav, sr = sf.read(path, dtype="float32")
            cached[chunk.index] = ChunkAudio(chunk.index, wav, sr, truncated=False)
        if cached:
            log.info("resuming with %d cached chunks", len(cached))
        return cached

    def _save_chunk(self, out_dir: Path, item: ChunkAudio):
        import soundfile as sf

        path = self._chunk_path(out_dir, item.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, item.wav, item.sample_rate, subtype="FLOAT")

    # --- generation + QA ---------------------------------------------------

    def _synthesize_chunks(
        self, out_dir: Path, chunks: List[Chunk], language: Optional[str] = None
    ) -> tuple[Dict[int, ChunkAudio], List[int], float]:
        audios = self._load_cached(out_dir, chunks)
        by_index = {c.index: c for c in chunks}

        # Shortest first, so each batch holds sentences of similar length. The
        # model decodes a whole batch until its LONGEST sentence is done, so a
        # 40-character sentence batched with a 250-character one sits finished
        # while the GPU keeps working for the long one. Measured on 30 real
        # chapters at batch size 8, document order spent 18,452 units of decode
        # work and length order 15,370 — 17% less, for nothing. Order of
        # generation is free to change: results are kept by chunk index, and
        # assembly walks the chunks in document order, not in this order.
        todo = sorted(
            (c for c in chunks if c.index not in audios), key=lambda c: len(c.text)
        )
        if todo:
            log.info("synthesizing %d chunks", len(todo))
            started = time.perf_counter()
            done = 0
            # Save as each batch lands. Anything already on disk survives a
            # crash, a kill, or a machine that runs out of memory.
            for item in self.synth.iter_synthesize(
                [c.text for c in todo], [c.index for c in todo], language=language
            ):
                audios[item.index] = item
                self._save_chunk(out_dir, item)
                done += 1
                if done % 10 == 0 or done == len(todo):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed else 0
                    remaining = (len(todo) - done) / rate if rate else 0
                    log.info(
                        "  %d/%d chunks (%.1f/min, ~%.0f min left)",
                        done,
                        len(todo),
                        rate * 60,
                        remaining / 60,
                    )

        if self.qa is None:
            return audios, [], 0.0

        pending = [c.index for c in chunks]
        worst = 0.0
        failed: List[int] = []

        for attempt in range(self.cfg.max_retries + 1):
            failed = []
            for index in pending:
                item = audios[index]
                result = self.qa.check(
                    index, item.wav, item.sample_rate, by_index[index].text
                )
                worst = max(worst, result.wer)
                if not result.passed or item.truncated:
                    failed.append(index)

            if not failed or attempt == self.cfg.max_retries:
                break

            log.info(
                "retry %d/%d for %d chunks",
                attempt + 1,
                self.cfg.max_retries,
                len(failed),
            )
            for item in self.synth.iter_synthesize(
                [by_index[i].text for i in failed],
                failed,
                seed=RETRY_SEED_BASE * (attempt + 1),
            ):
                audios[item.index] = item
                self._save_chunk(out_dir, item)
            pending = failed

        return audios, failed, worst

    # --- per-summary -------------------------------------------------------

    def render(self, job: SummaryJob) -> RenderResult:
        text = normalize(job.text)
        chunks = chunk_text(text, self.cfg.max_chunk_chars, self.cfg.heading_max_chars, self.cfg.one_sentence_per_chunk)
        if not chunks:
            raise ValueError(f"{job.id}: no text to synthesize after normalization")

        language = job.language or self.cfg.language
        render_id = self._render_id(text, language)
        out_dir = self.work_dir / job.id / render_id
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "audio.m4a"
        timing_path = out_dir / "timing.json"

        audio_key = timing_key = None
        if self.storage:
            audio_key = render_key(self.cfg.s3_prefix, job.id, render_id, "audio.m4a")
            timing_key = render_key(self.cfg.s3_prefix, job.id, render_id, "timing.json")

        # Already rendered and uploaded: nothing to do.
        if audio_path.exists() and (not self.storage or self.storage.exists(audio_key)):
            existing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
            log.info("%s already rendered (%s), skipping", job.id, render_id)
            return RenderResult(
                id=job.id,
                render_id=render_id,
                duration=existing.get("duration", 0.0),
                chunk_count=len(chunks),
                audio_path=audio_path,
                timing_path=timing_path,
                audio_key=audio_key,
                timing_key=timing_key,
                skipped=True,
            )

        log.info("%s: %d chunks (render %s)", job.id, len(chunks), render_id)
        audios, failed, worst_wer = self._synthesize_chunks(out_dir, chunks, language)

        wav, sample_rate, timings = audio_mod.assemble(
            list(audios.values()), chunks, self.cfg
        )
        wav = audio_mod.master(wav, sample_rate, self.cfg)
        duration = len(wav) / sample_rate

        audio_mod.encode(
            wav, sample_rate, audio_path, self.cfg, title=job.title, artist=job.author
        )
        audio_mod.write_timing_map(
            timings,
            timing_path,
            duration,
            metadata={
                "id": job.id,
                "title": job.title,
                "author": job.author,
                "render_id": render_id,
                "voice": {
                    "model": self.cfg.model_path,
                    "speaker": self.cfg.speaker,
                    "language": self.cfg.language,
                    "instruct": self.cfg.instruct,
                },
                "sample_rate": sample_rate,
            },
        )

        if self.storage:
            self.storage.upload(audio_path, audio_key)
            self.storage.upload(timing_path, timing_key)

        if failed:
            log.error(
                "%s: %d chunks still failing QA after %d retries: %s",
                job.id,
                len(failed),
                self.cfg.max_retries,
                failed,
            )

        return RenderResult(
            id=job.id,
            render_id=render_id,
            duration=duration,
            chunk_count=len(chunks),
            audio_path=audio_path,
            timing_path=timing_path,
            audio_key=audio_key,
            timing_key=timing_key,
            failed_chunks=failed,
            max_wer=worst_wer,
        )

    def run(self, jobs: List[SummaryJob]) -> List[RenderResult]:
        results: List[RenderResult] = []
        for i, job in enumerate(jobs, 1):
            log.info("[%d/%d] %s", i, len(jobs), job.id)
            try:
                results.append(self.render(job))
            except Exception:
                # One bad summary must not abandon the rest of a paid GPU session.
                log.exception("%s failed to render", job.id)
                if len(jobs) == 1:
                    raise
        return results


def audition(
    cfg: Config,
    text: str,
    speakers: List[str],
    out_dir: Path,
) -> List[dict]:
    """Render the same passage in several voices so you can pick one.

    Also reports the realtime factor per voice, which is the number that tells
    you how long a full catalog render will actually take on this machine.
    """
    import soundfile as sf

    normalized = normalize(text)
    chunks = chunk_text(normalized, cfg.max_chunk_chars, cfg.heading_max_chars, cfg.one_sentence_per_chunk)
    if not chunks:
        raise ValueError("no text to synthesize after normalization")

    out_dir.mkdir(parents=True, exist_ok=True)
    synth = Synthesizer(cfg)
    results: List[dict] = []

    # Load before timing starts. Otherwise the first voice absorbs the whole
    # model-load cost and its realtime factor is meaningless.
    load_started = time.perf_counter()
    synth.load()
    log.info("model loaded in %.1fs", time.perf_counter() - load_started)

    for speaker in speakers:
        log.info("auditioning %s (%d chunks)", speaker, len(chunks))
        started = time.perf_counter()
        audios = synth.synthesize(
            [c.text for c in chunks], [c.index for c in chunks], speaker=speaker
        )
        elapsed = time.perf_counter() - started

        wav, sample_rate, _ = audio_mod.assemble(audios, chunks, cfg)
        wav = audio_mod.master(wav, sample_rate, cfg)
        duration = len(wav) / sample_rate

        path = out_dir / f"{speaker.lower()}.wav"
        sf.write(path, wav, sample_rate)

        results.append(
            {
                "speaker": speaker,
                "path": str(path),
                "audio_seconds": round(duration, 2),
                "generation_seconds": round(elapsed, 2),
                "realtime_factor": round(duration / elapsed, 2) if elapsed else None,
            }
        )

    return results


def estimate(jobs: List[SummaryJob], cfg: Config) -> dict:
    """Dry-run: chunk everything and report the shape of the batch.

    Run this before starting a GPU so you know the chunk count and can catch
    normalization problems for free.
    """
    total_chunks = 0
    total_chars = 0
    oversized = 0
    per_job = []

    for job in jobs:
        text = normalize(job.text)
        chunks = chunk_text(text, cfg.max_chunk_chars, cfg.heading_max_chars, cfg.one_sentence_per_chunk)
        chars = sum(len(c.text) for c in chunks)
        total_chunks += len(chunks)
        total_chars += chars
        oversized += sum(1 for c in chunks if len(c.text) > cfg.max_chunk_chars)
        per_job.append(
            {"id": job.id, "chunks": len(chunks), "chars": chars}
        )

    # ~15 characters per second of speech at a normal narration pace.
    est_seconds = total_chars / 15.0
    return {
        "summaries": len(jobs),
        "chunks": total_chunks,
        "characters": total_chars,
        "oversized_chunks": oversized,
        "estimated_audio_hours": round(est_seconds / 3600.0, 2),
        "per_summary": per_job,
    }
