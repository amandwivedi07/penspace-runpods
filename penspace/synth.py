"""Batched Qwen3-TTS generation for Penspace narration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import runtime
from .config import Config

log = logging.getLogger(__name__)

# Apple Silicon shares memory with the OS, so an oversized batch swaps rather
# than failing. Keep it small unless we are on a discrete GPU.
MAX_UNIFIED_MEMORY_BATCH = 4


@dataclass
class ChunkAudio:
    index: int
    wav: np.ndarray
    sample_rate: int
    truncated: bool


class Synthesizer:
    """Wraps Qwen3TTSModel with the fixed narration voice and batching."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._torch = None
        self._device = None
        self._clone_prompt = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = runtime.resolve_device(self.cfg.device)
        return self._device

    def load(self):
        if self._model is not None:
            return
        # Imported lazily so the pure-python stages (normalize, chunk) stay
        # importable on machines without torch.
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        device = self.device
        dtype = runtime.resolve_dtype(self.cfg.dtype, device)
        attn = runtime.resolve_attn(self.cfg.attn_implementation, device)

        log.info(
            "Loading %s (%s)",
            self.cfg.active_model_path,
            runtime.describe(device, dtype, attn),
        )
        # torch 2.6 + accelerate device_map initializes on the meta device, then
        # .to(cuda) raises: "Cannot copy out of meta tensor". Load on CPU and
        # move the weights ourselves.
        self._model = Qwen3TTSModel.from_pretrained(
            self.cfg.active_model_path,
            dtype=dtype,
            attn_implementation=attn,
            low_cpu_mem_usage=False,
        )
        if device != "cpu":
            self._model.model.to(device)
            self._model.device = torch.device(device)

        if self.cfg.is_clone:
            if not self.cfg.ref_audio:
                raise ValueError("voice_mode='clone' requires ref_audio")
            return

        supported = self._model.get_supported_speakers()
        if supported and self.cfg.speaker.lower() not in {s.lower() for s in supported}:
            raise ValueError(
                f"speaker {self.cfg.speaker!r} not supported by this model; "
                f"available: {sorted(supported)}"
            )

    def clone_prompt(self):
        """Build the reference-voice prompt once and reuse it everywhere.

        Recomputing it per chunk would be both slower and less consistent; one
        frozen prompt keeps timbre identical across the whole catalog.
        """
        if self._clone_prompt is None:
            self.load()
            if not self.cfg.x_vector_only and not self.cfg.ref_text:
                raise ValueError(
                    "ICL cloning needs ref_text (the transcript of ref_audio). "
                    "Pass --ref-text, let penspace transcribe it, or set "
                    "--x-vector-only for lower-fidelity cloning."
                )
            log.info("building clone prompt from %s", self.cfg.ref_audio)
            self._clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=self.cfg.ref_audio,
                ref_text=None if self.cfg.x_vector_only else self.cfg.ref_text,
                x_vector_only_mode=self.cfg.x_vector_only,
            )
        return self._clone_prompt

    def _gen_kwargs(self) -> dict:
        return dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            temperature=self.cfg.temperature,
            repetition_penalty=self.cfg.repetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
        )

    def supported_speakers(self) -> List[str]:
        self.load()
        return sorted(self._model.get_supported_speakers() or [])

    def batch_size(self) -> int:
        """Cap the batch on unified-memory devices.

        A large batch multiplies KV-cache footprint, and on an Apple Silicon
        machine that shares memory with the OS the result is swap, not an OOM
        error -- the whole machine grinds instead of failing fast.
        """
        if self.device.startswith("cuda"):
            return self.cfg.batch_size
        capped = min(self.cfg.batch_size, MAX_UNIFIED_MEMORY_BATCH)
        if capped < self.cfg.batch_size:
            log.info(
                "capping batch size %d -> %d on %s",
                self.cfg.batch_size,
                capped,
                self.device,
            )
        return capped

    def iter_synthesize(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """Yield ChunkAudio as each batch completes.

        Streaming rather than returning a list is what makes the caller's chunk
        cache actually resumable: a crash costs the current batch, not every
        chunk generated so far.
        """
        self.load()
        voice = speaker or self.cfg.speaker
        # Per call, not per endpoint. The model already takes a language for
        # every item in the batch; only this layer was pinning it to the
        # deployed config, which meant one endpoint could narrate exactly one
        # language and five languages meant five deployments.
        lang = language or self.cfg.language
        if indices is None:
            indices = list(range(len(texts)))

        size = self.batch_size()
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            batch_idx = indices[start : start + size]
            n = len(batch)

            if seed is not None:
                # Vary per batch so a retry does not reproduce the same failure.
                self._torch.manual_seed(seed + start)

            if self.cfg.is_clone:
                # The Base model has no instruction control, so `instruct` is
                # deliberately not passed here.
                wavs, sr = self._model.generate_voice_clone(
                    text=batch,
                    language=[lang] * n,
                    voice_clone_prompt=self.clone_prompt(),
                    **self._gen_kwargs(),
                )
            else:
                wavs, sr = self._model.generate_custom_voice(
                    text=batch,
                    speaker=[voice] * n,
                    language=[lang] * n,
                    instruct=[self.cfg.instruct] * n,
                    **self._gen_kwargs(),
                )

            ceiling = self.cfg.max_chunk_seconds
            for idx, wav in zip(batch_idx, wavs):
                wav = np.asarray(wav, dtype=np.float32)
                duration = len(wav) / sr
                truncated = duration >= 0.98 * ceiling
                if truncated:
                    log.warning(
                        "chunk %d hit the %.0fs generation ceiling and is likely "
                        "truncated; reduce max_chunk_chars",
                        idx,
                        ceiling,
                    )
                yield ChunkAudio(
                    index=idx, wav=wav, sample_rate=sr, truncated=truncated
                )

    def synthesize(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
        speaker: Optional[str] = None,
    ) -> List[ChunkAudio]:
        """Collect every chunk. Prefer iter_synthesize when you can checkpoint."""
        return list(self.iter_synthesize(texts, indices, seed, speaker))
