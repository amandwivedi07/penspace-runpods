"""Cartesia Sonic as a synthesis backend.

A drop-in alternative to `synth.Synthesizer`: same `load()` /
`iter_synthesize()` / `synthesize()` contract, same `ChunkAudio` out, so the
chunker, QA gate, mastering, S3 upload and the manifest all carry over
unchanged. Only the thing that turns text into samples differs.

Why it exists
-------------
Qwen3-TTS on a rented GPU is dramatically cheaper per character and cannot
speak Hindi. Cartesia costs real money per character and can. Neither wins
outright, so the renderer supports both and the choice is made per run.

What differs from the local model
---------------------------------
- **Characters are money.** One credit per character, so a retry is not free
  the way a retry on your own GPU is. `characters_sent` tracks the spend and
  `estimate_credits()` prices a run before it starts.
- **Throughput is network-bound, not GPU-bound.** Chunks go out concurrently;
  `cartesia_concurrency` is the dial. There is no batching and no VRAM, so
  `batch_size` is ignored.
- **Nothing is truncated.** The local model caps at `max_new_tokens` codec
  frames and marks chunks it cut short; a hosted model has no such ceiling, so
  `truncated` is always False.
- **The voice is an id, not a recording.** Cloning happens in Cartesia's
  dashboard and yields a voice id; `ref_audio` plays no part here.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

from .config import Config
from .synth import ChunkAudio

log = logging.getLogger(__name__)

API_URL = "https://api.cartesia.ai/tts/bytes"
# Cartesia pins behaviour to a dated API version. It is sent on every request
# and belongs in the render fingerprint for the same reason a model path does.
API_VERSION = "2026-08-14"

# Raw float32 comes back as samples, not a container: no decoder, no ffmpeg
# round trip, and it is exactly what `audio.assemble` already expects.
ENCODING = "pcm_f32le"

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# The language codes Cartesia takes, keyed by the ones the manifest uses. Only
# a mapping where the two disagree is needed; anything absent passes through.
LANGUAGE_ALIASES = {
    "english": "en", "german": "de", "french": "fr", "spanish": "es",
    "japanese": "ja", "hindi": "hi", "portuguese": "pt", "chinese": "zh",
    "italian": "it", "korean": "ko",
}


def cartesia_language(name: Optional[str]) -> str:
    """Map a manifest language ("German", "de") to Cartesia's code."""
    if not name:
        return "en"
    key = str(name).strip().lower()
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    # Already a code, possibly regional ("en-GB"): hand it over as given.
    return key


class CartesiaError(RuntimeError):
    """A request failed in a way retrying will not fix."""


class CartesiaSynthesizer:
    """Synthesize chunks through Cartesia's HTTP API, concurrently."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._api_key: Optional[str] = None
        # Every character billed in this process. Printed at the end of a run,
        # because a spend you only discover on the invoice is not a budget.
        self.characters_sent = 0

    # -- lifecycle ---------------------------------------------------------

    def load(self):
        """Validate credentials and voice WITHOUT synthesizing anything.

        A warm-up call would cost credits, so the checks are the free ones:
        the key is present, a voice is named, and the voice actually resolves.
        """
        key = self.cfg.cartesia_api_key or os.environ.get("CARTESIA_API_KEY")
        if not key:
            raise CartesiaError(
                "No Cartesia API key. Set CARTESIA_API_KEY or pass --cartesia-key."
            )
        if not self.cfg.cartesia_voice_id:
            raise CartesiaError(
                "No Cartesia voice. Pass --voice-id, or set PENSPACE_CARTESIA_VOICE."
            )
        self._api_key = key

        # GET /voices/:id costs nothing and turns a typo into an error now
        # rather than 129 chunks into a run.
        req = urllib.request.Request(
            f"https://api.cartesia.ai/voices/{self.cfg.cartesia_voice_id}",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.cartesia_timeout) as resp:
                voice = json.loads(resp.read().decode("utf-8"))
            log.info(
                "Cartesia voice %s (%s), model %s",
                voice.get("name", "?"),
                self.cfg.cartesia_voice_id,
                self.cfg.cartesia_model,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (401, 403):
                raise CartesiaError(f"Cartesia rejected the API key: {body}") from exc
            if exc.code == 404:
                raise CartesiaError(
                    f"No such Cartesia voice: {self.cfg.cartesia_voice_id}"
                ) from exc
            # Anything else here is not worth blocking a run over — the first
            # synthesis call will surface a real problem clearly enough.
            log.warning("could not verify the voice (HTTP %s): %s", exc.code, body)
        except urllib.error.URLError as exc:
            log.warning("could not reach Cartesia to verify the voice: %s", exc)

    # -- pricing -----------------------------------------------------------

    @staticmethod
    def estimate_credits(texts: List[str]) -> int:
        """Credits a batch will cost. One credit per character, before retries."""
        return sum(len(t) for t in texts)

    # -- synthesis ---------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Cartesia-Version": API_VERSION,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _body(self, text: str, language: Optional[str]) -> bytes:
        payload = {
            "model_id": self.cfg.cartesia_model,
            "transcript": text,
            "voice": self.cfg.cartesia_voice_id,
            "output_format": {
                "container": "raw",
                "encoding": ENCODING,
                "sample_rate": self.cfg.cartesia_sample_rate,
            },
            # Exactly one of language/locale may be set.
            "language": cartesia_language(language or self.cfg.language),
            "generation_config": {"speed": self.cfg.cartesia_speed},
        }
        return json.dumps(payload).encode("utf-8")

    def _post(self, text: str, language: Optional[str]) -> np.ndarray:
        """One chunk, with backoff on the failures that are worth retrying."""
        body = self._body(text, language)
        delay = self.cfg.cartesia_backoff

        for attempt in range(self.cfg.cartesia_max_retries + 1):
            req = urllib.request.Request(
                API_URL, data=body, headers=self._headers(), method="POST"
            )
            try:
                with urllib.request.urlopen(
                    req, timeout=self.cfg.cartesia_timeout
                ) as resp:
                    raw = resp.read()
                # Billed on success. A failed request that never produced audio
                # is not counted, which is the honest reading of the invoice.
                self.characters_sent += len(text)
                return np.frombuffer(raw, dtype=np.float32).copy()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code not in RETRY_STATUS or attempt == self.cfg.cartesia_max_retries:
                    raise CartesiaError(
                        f"Cartesia HTTP {exc.code}: {detail}"
                    ) from exc
                # Honour Retry-After when the server sends one; it knows the
                # rate limit better than our backoff curve does.
                wait = float(exc.headers.get("Retry-After") or delay)
                log.warning(
                    "Cartesia HTTP %s, retrying in %.1fs (%d/%d)",
                    exc.code, wait, attempt + 1, self.cfg.cartesia_max_retries,
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.cfg.cartesia_max_retries:
                    raise CartesiaError(f"Cartesia unreachable: {exc}") from exc
                wait = delay
                log.warning("Cartesia unreachable (%s), retrying in %.1fs", exc, wait)

            time.sleep(wait)
            delay *= 2

        raise CartesiaError("unreachable")  # pragma: no cover — loop always returns

    def iter_synthesize(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
        seed: Optional[int] = None,          # noqa: ARG002 — no sampling to seed
        speaker: Optional[str] = None,       # noqa: ARG002 — the voice id is the speaker
        language: Optional[str] = None,
    ):
        """Yield ChunkAudio as each request lands, in completion order.

        The caller keys results by `index` and checkpoints each one, so finishing
        out of order is not only safe, it is what makes a long run resumable.
        """
        if self._api_key is None:
            self.load()
        idx = list(indices) if indices is not None else list(range(len(texts)))
        if len(idx) != len(texts):
            raise ValueError("texts and indices must be the same length")

        rate = self.cfg.cartesia_sample_rate
        workers = max(1, self.cfg.cartesia_concurrency)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._post, text, language): i
                for text, i in zip(texts, idx)
            }
            for future in as_completed(futures):
                i = futures[future]
                wav = future.result()  # a CartesiaError here fails the run
                yield ChunkAudio(index=i, wav=wav, sample_rate=rate, truncated=False)

    def synthesize(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
        seed: Optional[int] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[ChunkAudio]:
        """Collect every chunk. Prefer iter_synthesize when you can checkpoint."""
        return list(self.iter_synthesize(texts, indices, seed, speaker, language))
