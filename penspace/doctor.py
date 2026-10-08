"""Is this pod actually rendering on the GPU?

Written after a session where a pod reported `cuda available: True`, loaded a
1.9B model into 3.9 GB of VRAM, and then generated at 0% GPU and 130% CPU --
roughly a hundredth of the expected speed, while billing GPU rates. Nothing in
the normal logs said so. Everything *looked* right.

So this asks the question directly and answers it in under a minute:

    python -m penspace.cli doctor

It reports where the weights live, how much of the generate call was actually
spent in CUDA kernels, and the realtime factor -- the only number that decides
whether a GPU backfill is worth renting at all.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# Short enough to be cheap, long enough that start-up noise doesn't dominate.
PROBE_TEXT = (
    "Small habits compound over time into results that feel sudden only "
    "because the earlier gains were invisible."
)


def _device_report(model) -> dict:
    """Where the weights actually are, as opposed to where `.device` claims."""
    import torch.nn as nn

    by_device: dict[str, int] = {}
    for attr in vars(model).values():
        if not isinstance(attr, nn.Module):
            continue
        for param in attr.parameters():
            key = str(param.device)
            by_device[key] = by_device.get(key, 0) + param.numel()
    return by_device


def run(cfg) -> int:
    import torch

    from .synth import Synthesizer

    print("=" * 62)
    print("PENSPACE DOCTOR")
    print("=" * 62)

    try:
        import flash_attn

        flash = f"{flash_attn.__version__}"
    except Exception:  # noqa: BLE001 - absence is the finding, not an error
        flash = "MISSING -- the model falls back to its 'manual PyTorch' path"

    print(f"\ntorch            {torch.__version__}  (built for CUDA {torch.version.cuda})")
    print(f"flash-attn       {flash}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\nFAIL: no CUDA. Every render would run on the CPU.")
        return 1

    print(f"device           {torch.cuda.get_device_name(0)}")

    # A raw matmul separates "the GPU is broken" from "our code isn't using it".
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(50):
        x @ x
    torch.cuda.synchronize()
    tflops = 50 * 2 * 4096**3 / (time.perf_counter() - started) / 1e12
    print(f"raw matmul       {tflops:.1f} TFLOP/s")

    print("\nloading model...")
    synth = Synthesizer(cfg)
    load_started = time.perf_counter()
    synth.load()
    print(f"loaded in        {time.perf_counter() - load_started:.1f}s")

    print("\nWEIGHTS BY DEVICE")
    placement = _device_report(synth._model)
    # The decoder is not reachable through the model's parameters (it is not
    # a registered submodule), so it has to be reported on its own. Leaving it
    # out is what made an earlier run report "all weights on cuda:0" while a
    # 154M-parameter decoder ran on the CPU.
    _tok = getattr(getattr(synth._model, "model", None), "speech_tokenizer", None)
    if _tok is not None and getattr(_tok, "model", None) is not None:
        _p = next(_tok.model.parameters())
        _n = sum(p.numel() for p in _tok.model.parameters())
        print(f"  decoder      {_n / 1e6:8.1f}M params on {_p.device} ({_p.dtype})")
        if _p.device.type == "cpu" and synth.device != "cpu":
            print("  WARNING: the decoder runs on the CPU while the model is on "
                  f"{synth.device}. Every sentence is decoded there. "
                  "PENSPACE_DECODER_DEVICE=model moves it.")
    for device, count in sorted(placement.items(), key=lambda kv: -kv[1]):
        print(f"  {device:12s} {count / 1e6:8.1f}M params")
    on_cpu = sum(n for d, n in placement.items() if d.startswith("cpu"))
    if on_cpu:
        print(f"\n  WARNING: {on_cpu / 1e6:.1f}M parameters are on the CPU.")

    if cfg.is_clone:
        started = time.perf_counter()
        synth.clone_prompt()
        print(f"\nclone prompt     {time.perf_counter() - started:.1f}s (once per process)")

    # CUDA events time the GPU itself. Comparing that against wall clock is what
    # separates "slow GPU" from "not using the GPU" — the distinction no log
    # line in this pipeline was making.
    print("\nsynthesizing one probe chunk...")
    torch.cuda.synchronize()
    gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    gpu_start.record()
    chunks = synth.synthesize([PROBE_TEXT], [0])
    gpu_end.record()
    torch.cuda.synchronize()

    wall = time.perf_counter() - wall_started
    gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0
    audio_seconds = len(chunks[0].wav) / chunks[0].sample_rate

    print(f"\n  wall clock     {wall:.1f}s")
    print(f"  gpu time       {gpu_seconds:.1f}s  ({gpu_seconds / wall * 100:.0f}% of wall)")
    print(f"  audio produced {audio_seconds:.1f}s")
    print(f"  REALTIME       {audio_seconds / wall:.2f}x")

    print("\n" + "=" * 62)
    realtime = audio_seconds / wall
    if realtime < 1.0:
        print(f"VERDICT: {realtime:.2f}x realtime — too slow to be worth renting.")
        if flash.startswith("MISSING"):
            # Measured: 737s for 5.8s of audio on SDPA, against the same model
            # and GPU running normally with flash-attn. Check this first.
            print("\nflash-attn is MISSING, and that is almost certainly why.")
            print("Install the prebuilt wheel (see bootstrap.sh) and measure again")
            print("before concluding anything about this model or this GPU.")
        return 2
    if gpu_seconds / wall < 0.5:
        print("VERDICT: the GPU is barely involved. Most of the time is CPU work,")
        print("so renting a faster card will not help — find the CPU path first.")
        return 2
    print(f"VERDICT: rendering on the GPU. At {audio_seconds / wall:.1f}x realtime,")
    print(f"618 hours of audio would take about {618 / (audio_seconds / wall):.0f} GPU-hours.")
    return 0
