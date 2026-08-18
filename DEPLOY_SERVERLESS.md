# Deploying the narrator on RunPod Serverless

This is the serverless counterpart to `DEPLOY_RUNPOD.md`. Same renderer, same
`penspace/` package, same S3 layout — only the way it is invoked changes.

- **`DEPLOY_RUNPOD.md`** — a Pod you rent by the hour and feed a manifest.
  Right for a large one-off backfill.
- **this file** — an HTTP endpoint that renders a summary on demand and scales
  to zero. Right for the ongoing trickle: an editor publishes a summary, the
  backend asks for audio, nothing is rented in between.

Read §1 before you pick one. They are priced very differently and the backfill
is not the workload serverless is good at.

---

## 1. Read this before you spend anything

**A serverless GPU-second costs roughly 3× a community Pod GPU-second.** You pay
that premium for scale-to-zero, which is worth a great deal when the GPU would
otherwise sit idle and nothing at all when it would be busy for hours straight.

Two consequences:

1. **Do the backfill on a Pod.** It is hours of continuous GPU work with no idle
   time to eliminate. Serverless would charge the premium on every second of it
   and add a cold start per request on top.
2. **Run incremental generation here.** A handful of summaries a week is almost
   entirely idle time. A Pod would bill you around the clock to do minutes of
   work; this endpoint bills seconds.

Get the real number for your own catalogue before committing — the renderer
already knows how to tell you:

```bash
python -m penspace.cli estimate --manifest manifests/penspace.jsonl
```

Multiply the GPU hours it reports by the rate on the RunPod console (rates move;
do not trust a number written down in a repo, including this one).

> Your RunPod balance was **$7.75** when this was written. That funds testing
> this endpoint comfortably. It does not fund the full backfill — top up before
> starting that, or it will die partway and you will pay for the half that ran.

**Batch your requests.** The handler accepts a `jobs` array. Cold start — loading
Qwen3-TTS plus Whisper — is around a minute, and it is billed. Fifty summaries in
one request pay it once; fifty separate requests can pay it fifty times. This is
the single biggest lever on the bill.

---

## 2. Build and push the image

The weights are baked into the image on purpose. A batch Pod boots once, so
downloading ~7 GB from Hugging Face at boot is a rounding error; a serverless
worker boots every time the endpoint scales up, and you are billed for the wait.

```bash
docker build -t <dockerhub-user>/penspace-tts:1 -f penspace/Dockerfile.serverless .
docker push <dockerhub-user>/penspace-tts:1
```

Build on an x86 machine, or pass `--platform linux/amd64` from an Apple Silicon
Mac — a Mac's native arm64 image will not run on a RunPod GPU worker.

Expect ~15 GB and a slow first build (flash-attn compiles from source). Tag the
image with a version and bump it on every change: pointing an endpoint at
`:latest` makes it impossible to tell which workers are running which code.

---

## 3. Create the endpoint

RunPod console → **Serverless** → **New Endpoint** → your image tag.

| Setting | Value | Why |
|---|---|---|
| GPU | 24 GB (4090 / L4 / A5000) | The 1.7B model plus Whisper fits comfortably |
| Active workers | 0 | Anything above 0 bills continuously — that is a Pod with extra steps |
| Max workers | 2 to start | Raise once you have seen the bill |
| Idle timeout | 60–120 s | Long enough that a batched backlog reuses a warm worker |
| Execution timeout | Well above your longest batch | A killed request is billed and produces nothing |
| Container disk | 25 GB+ | Image plus `/work` scratch |

**Environment variables** — set these on the endpoint, not in the image, so keys
never live in a registry:

```
AWS_ACCESS_KEY_ID       (S3 write access to the bucket, nothing more)
AWS_SECRET_ACCESS_KEY
PENSPACE_S3_BUCKET      penspace-audio
PENSPACE_S3_REGION      ap-south-1
PENSPACE_REF_AUDIO      /app/penspace_ref.mp3
PENSPACE_REF_TEXT       "A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting, or receive criticism you secretly feared was true. The event may last only a few minutes, but the mind continues it for hours."
```

**`PENSPACE_REF_TEXT` is not optional here.** Left unset, each worker transcribes
the reference with Whisper at startup and that transcript feeds the render
fingerprint. Two workers that transcribe it even slightly differently produce
different `render_id`s for the same summary, the "already in S3" check stops
matching, and the endpoint quietly renders and bills everything twice. Pin it.
Same trap as `DEPLOY_RUNPOD.md` §4 — it just costs more here.

---

## 4. Call it

Single summary:

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"id": "happiness-trap", "title": "The Happiness Trap",
                 "author": "Russ Harris", "text": "..." }}'
```

A batch — this is the form to use for anything bulk:

```json
{"input": {"jobs": [
  {"id": "happiness-trap-ch1", "title": "…", "text": "…"},
  {"id": "happiness-trap-ch2", "title": "…", "text_url": "https://…/ch2.md"}
]}}
```

Each row needs an `id` and either `text` or `text_url` (http/https only).
`title` and `author` are optional. Use `/run` instead of `/runsync` for anything
that will take more than a few seconds, then poll `/status/<id>`.

Response:

```json
{"rendered": [{"id": "…", "render_id": "…", "duration": 1284.6,
               "chunk_count": 214,
               "audio_key": "penspace/audio/…/audio.m4a",
               "timing_key": "penspace/audio/…/timing.json",
               "failed_chunks": [], "max_wer": 0.021, "skipped": false}],
 "failed": [],
 "counts": {"rendered": 1, "failed": 0}}
```

`skipped: true` means that exact text and voice were already rendered and the
existing S3 object was returned — no GPU was used and nothing was billed for it.

`failed_chunks` is the field to check before publishing. A non-empty list means
the audio has silent gaps where those chunks should be: the file still plays, it
just drops a passage, which a spot-check will not catch. The backend's ingest
endpoint refuses such rows unless explicitly forced.

---

## 5. Failure semantics

The handler distinguishes three cases, because the caller should treat them
differently:

| What happened | Response | What the backend should do |
|---|---|---|
| Some rows rendered, some did not | 200, ids listed under `failed` | Retry only the listed ids |
| Every row failed | request FAILED | Retry the batch; alert if it fails again |
| Worker could not initialize | request FAILED | Alert — bad config or bad image, retrying will not help |

A malformed row (no `id`, unreachable `text_url`) is reported in `failed` rather
than raising, so one bad record cannot throw away the rest of a batch.

Retries are safe. `Runner.render` fingerprints the normalized text and the voice
into `render_id` and returns early when that object already exists in S3, so a
re-delivered request costs nothing. This is why pinning `PENSPACE_REF_TEXT`
matters so much: it is what keeps that fingerprint stable.

---

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| Every request pays a cold start | Requests arriving further apart than the idle timeout, or one-summary requests that should be batched |
| Same audio, new S3 keys each run | `PENSPACE_REF_TEXT` not pinned — §3 |
| Worker init fails on `S3Storage` | Bucket or AWS keys missing from the endpoint environment |
| `exec format error` in worker logs | Image built on Apple Silicon without `--platform linux/amd64` |
| CUDA OOM | Drop to a smaller GPU tier only after setting `PENSPACE_ATTN=sdpa`; flash-attn is what keeps VRAM low |
| Renders succeed but nothing in S3 | No `PENSPACE_S3_BUCKET` — audio was written to the worker's disk and lost on scale-down |
