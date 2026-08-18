# Penspace narration pipeline

Turns book-summary text into mastered, streamable audio in S3, using
Qwen3-TTS-12Hz-1.7B-CustomVoice.

Self-contained under `penspace/`: nothing in the upstream Qwen3-TTS tree is
modified, so this stays trivially rebaseable on upstream.

## Architecture

The app never talks to a GPU. Two systems, deliberately separate:

| | What | Where | Uptime |
|---|---|---|---|
| **Render** | text → chunks → audio → QA → master → S3 | rented GPU VM | hours per batch, then **off** |
| **Deliver** | presigned URL → player streams from S3 | your backend | always-on, ~free |

Why pre-generate: summaries are fixed content reused across users. Rendering
once and serving a static object is far cheaper and more reliable than
synthesizing per play, and it means the GPU is off almost all the time.

## Pipeline stages

1. **normalize** — strip markdown, drop citations, expand abbreviations, and
   spell out numbers, currency, years and percentages.
2. **chunk** — split at sentence boundaries into ~300-char pieces. The model
   emits at most `max_new_tokens` codec frames per call (~164 s of audio at the
   2048 default), so a 10-minute summary *cannot* be one generation. Small
   chunks also localize QA failures and make retries cheap.
3. **synthesize** — batched `generate_custom_voice` with a frozen speaker.
4. **QA gate** — transcribe every chunk back with Whisper and compare WER
   against the source. Failures regenerate with a new seed. Both sides pass
   through the same normalizer, since Whisper writes "2019" where the source
   says "twenty nineteen".
5. **master** — trim, join with paragraph-aware pauses, normalize to −16 LUFS
   (spoken-word standard), cap peaks, encode 64 kbps AAC with `+faststart`.
6. **upload** — content-addressed S3 keys plus a chunk timing map.

## Quick start

```bash
pip install -r penspace/requirements.txt

# Chunk the manifest and sanity-check normalization. No GPU, no model download.
python -m penspace.cli estimate --manifest penspace/manifest.example.jsonl

# Hear the same passage in each voice, and measure this machine's throughput.
python -m penspace.cli audition --speakers Ryan Aiden --out ./penspace_audition

# Render locally, no upload.
python -m penspace.cli render --manifest penspace/manifest.example.jsonl \
    --work-dir ./penspace_out

# Render and upload.
export AWS_REGION=ap-south-1
python -m penspace.cli render --manifest ./manifests/catalog.jsonl \
    --work-dir ./penspace_out --bucket penspace-audio
```

Always run `estimate` first. It catches normalization mistakes for free, before
you start paying for a GPU.

## Voice cloning

Pass `--ref-audio` and the pipeline switches to `Qwen3-TTS-12Hz-1.7B-Base` and
clones that voice instead of using a built-in speaker. Everything downstream --
chunking, QA, mastering, S3 -- is unchanged.

```bash
# Hear the clone before committing to it.
python -m penspace.cli audition --ref-audio ./narrator.wav \
    --text "Small habits compound into remarkable results."

# Render the catalog in the cloned voice.
python -m penspace.cli render --manifest ./manifests/catalog.jsonl \
    --ref-audio ./narrator.wav --bucket penspace-audio
```

**Only clone a voice you own or have written permission to use.** A cloned
narrator in a commercial app is a licensing question, not just a technical one.

Reference recording guidance:

- **3 s minimum**, 10-30 s is materially better.
- Clean speech: one speaker, no music, no background noise, no reverb.
- Any format soundfile reads; it is resampled to 24 kHz automatically.
- Natural reading pace in the style you want narrated — the clone picks up
  delivery, not just timbre.

The transcript of the reference is required for the high-quality ICL path. You
can pass it with `--ref-text`, but by default it is transcribed automatically
using the Whisper model already loaded for the QA gate. `--x-vector-only` skips
the transcript entirely and clones from the speaker embedding alone, which is
faster and lower fidelity.

The prompt is built **once** and reused across every chunk of every summary, so
timbre stays identical catalog-wide. Note the Base model has no instruction
control, so `--instruct` is ignored when cloning; delivery comes from the
reference recording instead.

Changing the reference recording changes the `render_id` (its bytes are hashed
in), so re-renders land on new S3 keys rather than serving stale audio.

## Ingesting source documents

Summaries usually arrive as `.docx`. `ingest` converts one and adds it to a
manifest:

```bash
python -m penspace.cli ingest "The Happiness Trap.docx" \
    --id happiness-trap --title "The Happiness Trap" --author "Russ Harris" \
    --manifest ./manifests/penspace.jsonl
```

Word files typically style every paragraph `Normal`, so there is no heading
information to read. Headings are recovered structurally instead: a short block
(≤80 chars, ≤12 words) that does not end in sentence punctuation is treated as a
section title and written as `## `. That matters for narration -- headings
become paragraph boundaries, and the chunker gives those a longer pause, so they
land as audible section breaks.

Check the recovered headings in the command's output before rendering. A missed
heading just reads as prose; a false positive puts an odd pause mid-paragraph.

`.md` and `.txt` are passed through unchanged.

## Manifest

JSONL, one summary per line. `id` is required plus either `text` or
`text_file`; `title` and `author` become m4a metadata.

```json
{"id": "atomic-habits", "title": "Atomic Habits", "author": "James Clear", "text": "..."}
{"id": "deep-work", "title": "Deep Work", "text_file": "summaries/deep-work.md"}
```

## Running on a Mac vs the GPU VM

Backends are auto-detected (`cuda` → `mps` → `cpu`) with a matching dtype and
attention implementation, so the same command works in both places.

- **Mac (MPS)**: float32 and SDPA attention. Good for developing the pipeline
  and auditioning voices; too slow for rendering a catalog. Whisper runs on CPU
  because CTranslate2 has no MPS backend. Needs `brew install ffmpeg sox`
  (qwen-tts shells out to the SoX binary and only warns when it is missing).
- **GPU VM (CUDA)**: bfloat16 and FlashAttention 2. This is where batches run.

Override with `PENSPACE_DEVICE`, `PENSPACE_DTYPE`, `PENSPACE_ATTN`.

## Output layout

```
{s3_prefix}/{summary_id}/{render_id}/audio.m4a
{s3_prefix}/{summary_id}/{render_id}/timing.json
```

`render_id` is a hash of the normalized text plus every voice setting. Change
the text or the voice and you get a new key, so `Cache-Control: immutable` is
safe and repeat plays cost almost nothing in CDN egress.

`timing.json` holds the start/end of every chunk. It costs nothing to produce
now and is what lets the app do sentence highlighting and scrub-to-paragraph
later without re-rendering the catalog.

`catalog.jsonl` in the work dir is the handoff to your app: one row per playable
summary with its keys and duration. Ingest it into your summaries table.

## App integration

`serve_urls.py` is a reference FastAPI service — fold it into your existing
backend and swap the JSONL catalog for your real table:

```
GET /summaries/{id}/audio   -> {"url": "<presigned>", "expires_in": 3600, "duration": 612.4}
GET /summaries/{id}/timing  -> {"url": "<presigned>", "expires_in": 3600}
```

Presigned URLs support HTTP range requests, so players seek without downloading
the whole file. Keep the bucket private and mint URLs behind your entitlement
check.

Bucket setup: block all public access, and if you front it with CloudFront use
an Origin Access Control rather than making objects public.

## Resumability

Every chunk is cached as a wav under
`{work_dir}/{id}/{render_id}/chunks/`. Re-running a manifest re-synthesizes only
missing chunks and skips summaries already uploaded. A crash mid-batch costs you
the current chunk, not the batch — which matters when the GPU bills per second.

## Tuning

| Knob | Default | Notes |
|---|---|---|
| `--speaker` | `Aiden` | `Ryan` is the other English voice. Pick one and freeze it. |
| `--instruct` | calm, measured | Applied to every chunk; keeps the read consistent. |
| `--max-chunk-chars` | 300 | Lower if chunks approach the token ceiling. |
| `--batch-size` | 8 | Raise on a big GPU for throughput. |
| `--max-wer` | 0.15 | Lower is stricter and costs more retries. |
| `--no-qa` | off | Only for quick iteration. Never for a real catalog. |
| `temperature` | 0.75 | Below the upstream examples' 0.9: long-form wants stability. |

Exit code `2` means some chunks never passed QA; the ids are printed to stderr.

## Tests

```bash
pytest penspace/tests -q
```

These pin real bugs found while building the pipeline — deleted numbers, fused
words, merged paragraphs. All are inaudible in the code and glaring in the
audio, so keep them green when touching `normalize.py` or `chunker.py`.

## Gotchas

- `transformers` is pinned to `4.57.3` upstream. Don't upgrade it.
- FlashAttention 2 builds from source and is CUDA-only. On failure, set
  `PENSPACE_ATTN=sdpa`.
- ffmpeg is required for encoding.
- Pre-download the weights onto the VM's persistent volume, or bake them into
  the image. Otherwise every boot burns paid GPU time on a download.
- Set an auto-shutdown on the VM. Per-second billing is only cheap if the box
  is actually off between batches.
