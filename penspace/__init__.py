"""Penspace book-summary narration pipeline built on Qwen3-TTS.

Text in, mastered streamable audio in S3 out. The GPU is only needed for the
`render` stage; everything else (normalization, chunking, presigned playback
URLs) runs anywhere.
"""

from .config import Config
from .chunker import Chunk, chunk_text
from .normalize import normalize
from .runner import Runner, SummaryJob, RenderResult, estimate, load_manifest
from .storage import S3Storage, render_key

__all__ = [
    "Config",
    "Chunk",
    "chunk_text",
    "normalize",
    "Runner",
    "SummaryJob",
    "RenderResult",
    "estimate",
    "load_manifest",
    "S3Storage",
    "render_key",
]
