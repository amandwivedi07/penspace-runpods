# Rendering through Cartesia

The third way to run the narrator, alongside `DEPLOY_RUNPOD.md` (a GPU you
rent) and `DEPLOY_SERVERLESS.md` (a GPU that scales to zero). Same chunker,
same mastering, same S3 layout, same manifest — only synthesis moves.

```bash
export CARTESIA_API_KEY=...
python -m penspace.cli render \
  --manifest manifests/atomic-habits.jsonl \
  --voice-id <your-voice-id> \
  --bucket penspace-media --no-qa
```

`--voice-id` implies `--backend cartesia`. There is no `--ref-audio` here: the
clone already happened in Cartesia's dashboard and the id names the result.

---

## 1. Price it before you run it

Characters are money on this backend — **1 credit per character** — so a
mistake is a bill rather than a wasted afternoon. `estimate` costs nothing and
needs no key:

```bash
python -m penspace.cli estimate --manifest manifests/atomic-habits.jsonl \
  --voice-id x --credit-rate 38
```

```json
{ "chunks": 129, "characters": 25505,
  "cartesia_credits": 25505, "cartesia_cost_usd": 0.97 }
```

`--credit-rate` is USD per million credits and defaults to 38, the Scale-plan
overage rate at the time of writing. **Rates move and plans differ — read yours
off the console.** For the whole catalogue:

| | characters | at $38/M |
|---|---|---|
| English | 19.6M | ~$740 |
| Five languages (no Hindi) | 31.1M | ~$1,180 |
| All six | 34.2M | ~$1,300 |

For comparison, the same backfill on a community Pod is $10–40. **Cartesia is
not the cheap option and is not meant to be.** It is the option that needs no
GPU, no volume, no QA gate — and the only one that can speak **Hindi**, which
Qwen3-TTS cannot do at all.

`--max-credits N` refuses to start a run that would exceed N, which is the
cheap insurance against pointing a paid backend at the wrong manifest.

## 2. What it cannot spend twice

A render is cached under a `render_id` fingerprinted from the text, the voice
and **the backend**. Three consequences, all of them financial:

- **Re-running a finished manifest bills nothing.** Chunks already on disk or
  in S3 are not re-requested. Resume freely after a crash.
- **Switching from Qwen to Cartesia re-renders everything**, deliberately. They
  are different recordings, and an id that could not tell them apart would
  serve the old audio forever.
- **Switching voices re-renders everything too.** Same reason.

The number printed at the end of a run — `cartesia credits spent` — counts
requests that actually returned audio, so it includes QA retries and excludes
skips. Prefer it to the manifest's character count when reconciling the bill.

## 3. Settings that matter

| Flag / env | Default | Why you would change it |
|---|---|---|
| `--cartesia-concurrency` / `PENSPACE_CARTESIA_CONCURRENCY` | 8 | Throughput is network-bound here, not GPU-bound. This is what `batch_size` is for the local model; raise it until you see 429s. |
| `--cartesia-model` / `PENSPACE_CARTESIA_MODEL` | `sonic-3.6` | Pin a model so the voice cannot drift mid-catalogue. |
| `--no-qa` | off | The ASR gate runs Whisper **locally**. With no GPU it will dominate the wall clock, and a hosted model is a much weaker case for it. Leave it on for a first book, then decide. |
| `--max-credits` | none | Refuse to start an unexpectedly large run. |

Rate limits are retried with backoff, honouring `Retry-After`. A 4xx that is
not a rate limit fails immediately with the server's message — retrying a
malformed request just spends longer being wrong.

## 4. Languages

The job's language is sent per request, taken from the manifest — not from the
endpoint's default. `"German"` and `"de"` both resolve. Supported here:
`en de fr es ja hi pt zh it ko` and regional variants such as `en-GB`.

Hindi is the reason this backend exists. Qwen3-TTS cannot produce it, which
leaves 1,059 chapters and roughly 60 hours of audio that have no other route.

## 5. The split worth considering

Backfill English and the four European/Japanese languages on a Pod, where the
whole job is $10–40. Use Cartesia for the two things the Pod cannot do: Hindi,
and every new summary from here on. That is the full six-language catalogue for
roughly a tenth of doing all of it here.
