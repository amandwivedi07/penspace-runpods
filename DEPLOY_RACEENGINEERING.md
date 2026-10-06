# Deploying Penspace rendering on raceengineering.ai

How to render the Penspace book-summary catalog on a rented GPU pod, in the
narrator voice cloned from `julian_ref_clean.mp3`.

Rendering is a **batch job, not a service**. The pod is on for the length of a
batch and off the rest of the time; the app only ever reads finished audio from
S3. Billing is per-minute, so an idle pod is pure waste.

---

## 0. Before you start

- Wallet topped up — **minimum ₹500**.
- SSH public key added under profile settings:

  ```bash
  ssh-keygen -t ed25519
  cat ~/.ssh/id_ed25519.pub     # paste into the profile page
  ```

- The deployment bundle `penspace-deploy.tgz`, containing the pipeline, your
  reference recording, the summary texts and the manifests.

---

## 1. Choose the GPU

| GPU | VRAM | ₹/hr | Verdict |
|---|---|---|---|
| **RTX 4090** | 24 GB | **₹50** | **Recommended.** Ada, best price/perf here. |
| RTX 3090 | 24 GB | ₹37 | Cheaper and slower. Fine if throughput doesn't matter. |
| L40S | 48 GB | ₹113 | Unnecessary. |
| A100 / H100 | 80 GB | ₹120–225 | Wildly oversized — you pay for VRAM you cannot use. |

Qwen3-TTS-12Hz-1.7B is ~3.4 GB in bf16. A 4090's 24 GB is already generous;
the larger cards buy nothing for this workload.

**Launch settings**

| Setting | Value |
|---|---|
| Template | PyTorch |
| Container disk | 60 GB (≈7 GB weights + audio + headroom) |

---

## 2. Upload and bootstrap

Take the SSH command from the pod's **Endpoints** panel.

```bash
scp -P <port> penspace-deploy.tgz root@<host>:/workspace/
ssh -p <port> root@<host>

cd /workspace && tar xzf penspace-deploy.tgz
bash penspace/bootstrap.sh
```

`bootstrap.sh` is idempotent and takes ~10 minutes the first time. It:

1. verifies the GPU with `nvidia-smi`
2. installs `ffmpeg`, `sox`, `libsndfile1`
3. creates a venv at `/workspace/venv` and installs the pinned stack
4. builds FlashAttention 2, falling back to SDPA if the build fails
5. pre-downloads all ~7 GB of weights into `/workspace/hf-cache`

Everything expensive lands in `/workspace` on purpose — see §5.

---

## 3. Pin the voice

Add to `~/.bashrc` on the pod so every render uses the identical narrator:

```bash
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf-cache
export PENSPACE_REF_AUDIO=/workspace/julian_ref_clean.mp3
export PENSPACE_REF_TEXT="A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting, or receive criticism you secretly feared was true. The event may last only a few minutes, but the mind continues it for hours."
```

Setting `PENSPACE_REF_AUDIO` switches the pipeline to the Base checkpoint and
clone mode automatically.

**Pin `PENSPACE_REF_TEXT` rather than letting it auto-transcribe.** The render
id is a hash of the text plus every voice setting, including this transcript.
Whisper runs at a different precision on CUDA than on Apple Silicon, so an
auto-transcribed reference could differ by a word and silently produce a
different render id — new S3 keys for audio that is otherwise identical.

CUDA settings resolve themselves: bf16, FlashAttention 2, and the batch size
un-caps from the Apple-Silicon limit of 4 back to 8.

---

## 4. Render

Always dry-run first. `estimate` needs no GPU and catches normalization
problems for free:

```bash
python -m penspace.cli estimate --manifest manifests/rest.jsonl
```

Then time a single book. **This number sizes everything else.**

```bash
time python -m penspace.cli render \
    --manifest manifests/rest.jsonl \
    --work-dir /workspace/out
```

Then the full catalog, straight to S3:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=ap-south-1

python -m penspace.cli render \
    --manifest manifests/penspace.jsonl \
    --work-dir /workspace/out \
    --bucket penspace-audio
```

Exit code `2` means some chunks never passed the QA gate; the failing ids print
to stderr. Everything else still rendered.

### Ingesting new summaries

```bash
python -m penspace.cli ingest "The Happiness Trap.docx" \
    --id happiness-trap --title "The Happiness Trap" --author "Russ Harris" \
    --manifest manifests/penspace.jsonl
```

Check the recovered headings in the output before rendering — Word files carry
no usable heading styles, so they are inferred structurally.

---

## 5. The rule that will cost you a re-render

**Pause preserves `/workspace`. Terminate deletes it permanently.**

So:

- Keep the venv, weights cache and outputs under `/workspace` (bootstrap does).
- **Upload to S3 before terminating.** Audio that exists only on the pod is one
  click from gone.
- **The SSH port changes on every resume.** Re-read the Endpoints panel; do not
  reuse the previous command.

Renders are resumable at chunk granularity — completed chunks are cached under
`/workspace/out/<id>/<render_id>/chunks/`, and re-running a manifest
re-synthesizes only what is missing and skips summaries already uploaded. A
crash or a pause costs one batch, not the run.

---

## 6. Operating loop

```
launch pod → bootstrap (first time only) → sync manifest
   → estimate → render → verify QA → upload to S3 → PAUSE the pod
```

Pause between batches. Terminate only when you are done for good and everything
is in S3.

Rough projection for ~500 titles (~167 hours of audio): at an assumed 10×
realtime that is ~17 GPU-hours, about **₹850 + 18% GST**. The 10× is a guess —
replace it with the measurement from §4 and scale linearly.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `flash-attn` build fails | Harmless. `export PENSPACE_ATTN=sdpa`. |
| CUDA OOM | Lower `--batch-size` (default 8). |
| Chunks hit the generation ceiling | Lower `--max-chunk-chars`; the model caps at ~164 s of audio per call. |
| Exit code 2 | Chunks failed QA after retries. Inspect the printed ids; raise `--max-wer` only if the audio actually sounds fine. |
| Re-download on every resume | `HF_HOME` is not under `/workspace`. |
| Same audio, new S3 keys | `PENSPACE_REF_TEXT` was not pinned — see §3. |
| `sox: command not found` | Bootstrap didn't run, or ran without apt. |

---

## 8. After rendering

`--bucket` writes:

```
{prefix}/{summary_id}/{render_id}/audio.m4a
{prefix}/{summary_id}/{render_id}/timing.json
```

and `/workspace/out/catalog.jsonl` — one row per playable summary with its keys
and duration. That file is the handoff to the app: ingest it into your
summaries table.

Keep the bucket private and mint presigned URLs behind your entitlement check;
`penspace/serve_urls.py` is a reference implementation. Presigned URLs support
HTTP range requests, so players seek without downloading the whole file.
