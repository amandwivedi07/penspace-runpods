"""Backend selection so the same pipeline runs on a rented GPU and on a Mac.

The upstream examples hardcode `cuda:0`, `bfloat16` and `flash_attention_2`.
None of those are right on Apple Silicon, so the choice is resolved at load
time instead. Everything defaults to "auto"; set the matching PENSPACE_* env
var to override.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    if preference and preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_attn(preference: str, device: str) -> str:
    """FlashAttention 2 needs CUDA *and* the flash_attn package.

    Being on CUDA is not enough: the Docker image ships flash_attn, but a
    bare-metal pod built from a plain PyTorch base does not. Asking for
    flash_attention_2 without the package raises ImportError deep inside
    transformers, after the model download and several minutes of import —
    so probe for it here and fall back rather than failing late.
    """
    if preference and preference != "auto":
        return preference
    if not device.startswith("cuda"):
        return "sdpa"
    import importlib.util

    if importlib.util.find_spec("flash_attn") is None:
        log.info("flash_attn not installed; using sdpa (slower, install flash-attn to speed up)")
        return "sdpa"
    return "flash_attention_2"


def resolve_dtype(preference: str, device: str):
    import torch

    if preference and preference != "auto":
        return getattr(torch, preference)
    if device.startswith("cuda"):
        return torch.bfloat16
    # MPS has patchy bfloat16 op coverage. float32 costs ~7GB for the 1.7B
    # model, which fits comfortably in unified memory; set PENSPACE_DTYPE=
    # float16 to trade a little stability for speed.
    return torch.float32


def whisper_device(device: str) -> tuple[str, str]:
    """faster-whisper runs on CTranslate2, which supports CPU and CUDA only."""
    if device.startswith("cuda"):
        return "cuda", "float16"
    return "cpu", "int8"


def describe(device: str, dtype, attn: str) -> str:
    return f"device={device} dtype={str(dtype).replace('torch.', '')} attn={attn}"
