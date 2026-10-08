"""Where does a render's time go, and what does moving the decoder change?

Run on a GPU pod, with the pod's own configuration (voice, reference text):

    python -m penspace.profile_decode            # full run, ~10-15 minutes
    PROFILE_QUICK=1 python -m penspace.profile_decode   # smoke test

Every configuration renders the SAME 32 sentences, drawn from real chapters
(Evicted; Everyone Communicates, Few Connect; Easy Riders, Raging Bulls) to
match the catalogue's length mix, and splits wall time into the language
model generating codes and the speech tokenizer decoding them to audio.

Configurations:
  A  decoder on CPU, as loaded             <- what production does today
  B  decoder on CPU in float32             <- isolates the bfloat16-on-CPU cost
  C  decoder on the GPU, batch 4/8/16/32   <- the candidate fix, and batch sizes

Results print as a table and are written to /work/profile_decode.json.
Nothing here changes how the worker renders.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

SENTENCES = [
    "People often trust the larger pattern more than the sentence because the pattern feels harder to fake.",
    "For tenants, however, the experience may be confusing and frightening.",
    "They learn to build connection even when natural chemistry is absent.",
    "For Biskind, the emergence of such films helps explain why the period became artistically significant: filmmakers were given opportunities to depict a complicated America at a moment when its familiar stories were increasingly difficult to believe.",
    "Housing is an essential expense, but its cost can consume such a large share of household income that meeting other basic needs becomes nearly impossible.",
    "Yet Sherrena's business generates considerable income.",
    "Many saw little connection between their experiences and the entertainment Hollywood continued to produce.",
    "For Arleen, finding another apartment after an eviction is an exhausting process.",
    "Maxwell argues that effective communicators increase their effort as the distance between themselves and the audience grows.",
    "Paramount executives had their own ideas about the film.",
    "Lamar's effort demonstrates considerable determination, but determination cannot compensate for an arrangement in which his expenses regularly exceed the resources available to meet them.",
    "That makes communication less about performance and more about responsibility.",
    "Desmond proposes making housing assistance available to all eligible low-income families rather than treating it as a limited benefit that only some receive.",
    "Financial insecurity leads to displacement, and displacement creates additional financial insecurity.",
    "Tobin sometimes offers tenants a trailer without charging for the structure itself, while continuing to collect rent for the land beneath it.",
    "They also evaluate the person delivering it.",
    "People cannot act on a message they cannot recall.",
    "Sherrena Tarver, a former schoolteacher, has discovered that renting properties in Milwaukee's poorest neighborhoods can be a profitable business.",
    "Yet information can travel perfectly from one person to another without creating trust, understanding, or influence.",
    "Economic hardship does not mean that money is absent from poor neighborhoods.",
    "The revolution began when the industry's declining confidence created room for artists whose understanding of American culture seemed more relevant than the assumptions of the executives employing them.",
    "When listeners understand something well enough to explain it to someone else, the communication has become more durable.",
    "Desmond places these personal struggles within Milwaukee's broader economic history.",
    "The same refusal to obey conventional rules that made the production distinctive could also make collaboration unstable.",
    "Under Milwaukee's nuisance property ordinance, landlords can face penalties when police are repeatedly called to their buildings.",
    "The question his investigation ultimately raises is whether a society can meaningfully address poverty while allowing its poorest families to remain perpetually at risk of losing their homes.",
    "In *Taxi Driver*, released in 1976, Martin Scorsese presents a New York City filled with loneliness, violence, sexual exploitation, and deep social alienation.",
    "Biskind's account reveals that New Hollywood was never simply a community of rebels united against an outdated industry.",
    "The availability of inexpensive properties does not necessarily translate into affordable homes for tenants living on extremely limited incomes.",
    "For a brief period, the people once considered too unconventional to represent mainstream American cinema became precisely the people Hollywood wanted to hire.",
    "His determination to cast Marlon Brando as Vito Corleone and Al Pacino as Michael Corleone created serious disagreements with studio executives.",
    "A communicator who begins from shared experience can then lead listeners toward a different perspective without making them feel abandoned along the way."
]

PRICE_PER_HOUR = float(os.environ.get("PROFILE_PRICE", "0.74"))


class _GpuSampler(threading.Thread):
    """nvidia-smi every 0.5s. Power is the honest signal: utilisation is a
    point sample that reads 0% between short kernel bursts."""

    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.power, self._halt = [], [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                u, p = (float(x) for x in out.split(","))
                self.util.append(u); self.power.append(p)
            except Exception:
                return
            self._halt.wait(0.5)

    def stop(self):
        self._halt.set(); self.join(timeout=2)
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
        return avg(self.util), avg(self.power)


def _cpu_times():
    try:
        f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        v = [int(x) for x in f]
        return sum(v), v[3] + v[4]          # total, idle + iowait
    except Exception:
        return None


def _cpu_busy(a, b):
    if not a or not b or b[0] == a[0]:
        return None
    return round(100 * (1 - (b[1] - a[1]) / (b[0] - a[0])), 1)


def main() -> int:
    import torch
    from .config import Config
    from .synth import Synthesizer

    quick = os.environ.get("PROFILE_QUICK") == "1"
    texts = SENTENCES[:3] if quick else SENTENCES
    chars = sum(len(t) for t in texts)

    cfg = Config.from_env()
    synth = Synthesizer(cfg)
    synth.load()
    synth.clone_prompt()
    gpu = synth.device
    on_gpu = gpu.startswith("cuda")
    sync = (lambda: torch.cuda.synchronize()) if on_gpu else (lambda: None)

    tok = synth._model.model.speech_tokenizer
    loaded = next(tok.model.parameters())
    print(f"model on {gpu}; decoder loaded on {loaded.device} as {loaded.dtype}")
    print(f"{len(texts)} sentences, {chars} characters\n")

    spent = {"decode": 0.0}
    real_decode = tok.decode

    captured = {}

    def timed_decode(*a, **k):
        # Keep the first batch of codes, on the CPU, for the parity check.
        if "codes" not in captured and a:
            captured["codes"] = [
                {kk: (vv.detach().cpu().clone() if hasattr(vv, "detach") else vv)
                 for kk, vv in d.items()} for d in a[0]]
        sync(); t = time.perf_counter()
        out = real_decode(*a, **k)
        sync(); spent["decode"] += time.perf_counter() - t
        return out

    tok.decode = timed_decode

    def place(device, dtype):
        tok.model.to(device=device, dtype=dtype)
        tok.device = torch.device(device)

    def run(name, batch):
        synth.cfg.batch_size = batch
        # Warm up on this placement and batch so first-call costs are excluded.
        list(synth.iter_synthesize(texts[:min(batch, len(texts))], language=cfg.language))
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        spent["decode"] = 0.0
        sampler = _GpuSampler() if on_gpu else None
        if sampler: sampler.start()
        c0 = _cpu_times(); sync(); t0 = time.perf_counter()
        try:
            items = list(synth.iter_synthesize(texts, language=cfg.language))
            error = None
        except Exception as exc:                     # e.g. CUDA OOM at batch 32
            items, error = [], f"{type(exc).__name__}: {str(exc)[:120]}"
        sync(); wall = time.perf_counter() - t0; c1 = _cpu_times()
        util, power = sampler.stop() if sampler else (None, None)
        audio = sum(len(i.wav) / i.sample_rate for i in items)
        row = {
            "config": name, "batch": batch, "error": error,
            "wall_s": round(wall, 2),
            "decode_s": round(spent["decode"], 2),
            "generate_s": round(wall - spent["decode"], 2),
            "decode_share_pct": round(100 * spent["decode"] / wall, 1) if wall else None,
            "chars_per_s": round(chars / wall, 1) if items else None,
            "audio_s": round(audio, 1),
            "realtime_x": round(audio / wall, 2) if items else None,
            "gpu_util_pct": util, "gpu_power_w": power,
            "cpu_busy_pct": _cpu_busy(c0, c1),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if on_gpu else None,
        }
        if items:
            row["usd_per_1m_chars"] = round(PRICE_PER_HOUR * (1e6 / row["chars_per_s"]) / 3600, 2)
            row["usd_per_audio_min"] = round(PRICE_PER_HOUR * (wall / 3600) / (audio / 60), 4)
        print(f"{name:34s} b={batch:<3d} "
              + (f"FAILED {error}" if error else
                 f"{row['chars_per_s']:6.1f} ch/s  decode {row['decode_share_pct']:5.1f}%  "
                 f"gpu {util}% / {power}W  cpu {row['cpu_busy_pct']}%  "
                 f"vram {row['vram_peak_gb']}GB  ${row['usd_per_1m_chars']}/1M ch"), flush=True)
        return row

    rows = []
    rows.append(run("A decoder CPU, as loaded (today)", 8))
    if not quick:
        place("cpu", torch.float32)
        rows.append(run("B decoder CPU, float32", 8))
    if on_gpu:
        place(gpu, loaded.dtype)
        for b in ((2,) if quick else (4, 8, 16, 32)):
            rows.append(run("C decoder on GPU", b))

    # PARITY: the same codes decoded where production decodes them today and
    # on the GPU. Moving the decoder must not change what a listener hears.
    parity = None
    if on_gpu and captured.get("codes"):
        def decode_on(device, dtype):
            place(device, dtype)
            wavs, _ = real_decode(captured["codes"])
            return [torch.as_tensor(w, dtype=torch.float64).flatten() for w in wavs]
        ref = decode_on("cpu", loaded.dtype)
        new = decode_on(gpu, loaded.dtype)
        worst_corr, worst_snr = 1.0, float("inf")
        for x, y in zip(ref, new):
            n = min(len(x), len(y)); x, y = x[:n], y[:n]
            corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
            noise = float(((x - y) ** 2).mean()); sig = float((x ** 2).mean())
            snr = 10 * __import__("math").log10(sig / noise) if noise > 0 else float("inf")
            worst_corr, worst_snr = min(worst_corr, corr), min(worst_snr, snr)
        parity = {"clips": len(ref), "min_correlation": round(worst_corr, 5),
                  "min_snr_db": round(worst_snr, 1)}
        verdict = "SAME AUDIO" if worst_corr > 0.995 else "DIFFERS — listen before switching"
        print(f"\nparity, GPU vs today's CPU decode on identical codes: "
              f"min correlation {worst_corr:.5f}, min SNR {worst_snr:.1f} dB -> {verdict}")

    out = Path(os.environ.get("PENSPACE_WORK_DIR", "/work")) / "profile_decode.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"price_per_hour": PRICE_PER_HOUR, "chars": chars, "rows": rows, "parity": parity}, indent=2))
        print(f"\nwritten: {out}")
    except OSError as exc:
        print(f"\ncould not write {out}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
