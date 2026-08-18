"""Configuration for the Penspace book-summary narration pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

# Fixed properties of Qwen3-TTS-Tokenizer-12Hz. See
# qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py
SAMPLE_RATE = 24000
DECODE_UPSAMPLE_RATE = 1920
CODEC_HZ = SAMPLE_RATE / DECODE_UPSAMPLE_RATE  # 12.5 frames per second of audio


def _env_str(key: str, default):
    return os.environ.get(key, default)


def _env_int(key: str, default):
    v = os.environ.get(key)
    return int(v) if v else default


def _env_float(key: str, default):
    v = os.environ.get(key)
    return float(v) if v else default


def _env_bool(key: str, default):
    v = os.environ.get(key)
    return v.lower() in ("1", "true", "yes") if v else default


@dataclass
class Config:
    # --- model ---
    # "custom_voice" uses the built-in speakers; "clone" reproduces a voice from
    # a reference recording using the Base model.
    voice_mode: str = "custom_voice"
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    clone_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    # --- voice clone (voice_mode="clone") ---
    # Path, URL or base64 of the reference recording, plus its transcript. The
    # transcript is required for ICL mode, which is the higher-quality path;
    # penspace transcribes it with Whisper automatically when not supplied.
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    # Speaker-embedding-only cloning: no transcript needed, lower fidelity.
    x_vector_only: bool = False

    speaker: str = "Aiden"
    language: str = "English"
    # Applied to every chunk so the narration voice stays consistent catalog-wide.
    instruct: str = "Calm, warm, measured narration at an unhurried pace."
    # All "auto": cuda -> mps -> cpu, with a matching dtype and attention
    # backend. See runtime.py.
    device: str = "auto"
    attn_implementation: str = "auto"
    dtype: str = "auto"

    # --- generation ---
    batch_size: int = 8
    max_new_tokens: int = 2048
    # Lower than the repo examples' 0.9: long-form narration wants stability
    # over expressiveness.
    temperature: float = 0.75
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    # --- chunking ---
    # The model caps at max_new_tokens codec frames (~164s at 2048). We stay far
    # under that: smaller chunks give finer QA granularity and cheaper retries.
    max_chunk_chars: int = 300
    sentence_gap_ms: int = 120
    paragraph_gap_ms: int = 350
    lead_in_ms: int = 200
    tail_ms: int = 400

    # --- QA gate ---
    qa_enabled: bool = True
    whisper_model: str = "small.en"
    max_wer: float = 0.15
    max_retries: int = 2

    # --- mastering ---
    target_lufs: float = -16.0  # spoken-word / podcast standard
    peak_ceiling_db: float = -1.0
    bitrate: str = "64k"

    # --- storage ---
    s3_bucket: Optional[str] = None
    s3_prefix: str = "penspace/audio"
    s3_region: Optional[str] = None
    presign_ttl: int = 3600
    # Object keys are content-addressed by summary id + revision, so immutable
    # caching is safe and keeps CDN egress near zero on repeat plays.
    cache_control: str = "public, max-age=31536000, immutable"

    @property
    def is_clone(self) -> bool:
        return self.voice_mode == "clone"

    @property
    def active_model_path(self) -> str:
        """Cloning needs the Base checkpoint; built-in speakers need CustomVoice."""
        return self.clone_model_path if self.is_clone else self.model_path

    @property
    def max_chunk_seconds(self) -> float:
        """Audio ceiling implied by max_new_tokens. Chunks near this are truncated."""
        return self.max_new_tokens / CODEC_HZ

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.model_path = _env_str("PENSPACE_MODEL", c.model_path)
        c.speaker = _env_str("PENSPACE_SPEAKER", c.speaker)
        c.language = _env_str("PENSPACE_LANGUAGE", c.language)
        c.instruct = _env_str("PENSPACE_INSTRUCT", c.instruct)
        c.device = _env_str("PENSPACE_DEVICE", c.device)
        c.attn_implementation = _env_str("PENSPACE_ATTN", c.attn_implementation)
        c.dtype = _env_str("PENSPACE_DTYPE", c.dtype)
        # Pinning the reference transcript keeps render ids stable across
        # machines: Whisper runs at a different precision on CUDA than on MPS,
        # and a differing transcript would change the fingerprint.
        c.ref_audio = _env_str("PENSPACE_REF_AUDIO", c.ref_audio)
        c.ref_text = _env_str("PENSPACE_REF_TEXT", c.ref_text)
        if c.ref_audio:
            c.voice_mode = "clone"
        c.batch_size = _env_int("PENSPACE_BATCH_SIZE", c.batch_size)
        c.max_new_tokens = _env_int("PENSPACE_MAX_NEW_TOKENS", c.max_new_tokens)
        c.temperature = _env_float("PENSPACE_TEMPERATURE", c.temperature)
        c.max_chunk_chars = _env_int("PENSPACE_MAX_CHUNK_CHARS", c.max_chunk_chars)
        c.qa_enabled = _env_bool("PENSPACE_QA", c.qa_enabled)
        c.whisper_model = _env_str("PENSPACE_WHISPER_MODEL", c.whisper_model)
        c.max_wer = _env_float("PENSPACE_MAX_WER", c.max_wer)
        c.max_retries = _env_int("PENSPACE_MAX_RETRIES", c.max_retries)
        c.target_lufs = _env_float("PENSPACE_TARGET_LUFS", c.target_lufs)
        c.bitrate = _env_str("PENSPACE_BITRATE", c.bitrate)
        c.s3_bucket = _env_str("PENSPACE_S3_BUCKET", c.s3_bucket)
        c.s3_prefix = _env_str("PENSPACE_S3_PREFIX", c.s3_prefix)
        c.s3_region = _env_str("AWS_REGION", c.s3_region)
        c.presign_ttl = _env_int("PENSPACE_PRESIGN_TTL", c.presign_ttl)
        return c

    def to_dict(self) -> dict:
        return asdict(self)
