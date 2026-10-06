# Deploying Penspace rendering on RunPod

Alternative to [DEPLOY_RACEENGINEERING.md](DEPLOY_RACEENGINEERING.md). Same
pipeline, same `bootstrap.sh`, same `/workspace` convention — only the
provisioning differs.

Rendering is a **batch job, not a service**. The pod runs for the length of a
batch and is stopped the rest of the time; the app only ever reads finished
audio from S3.

---

## Why RunPod might be the better fit

**Network Volumes.** This is the real differentiator. A Network Volume is
persistent storage that lives independently of any pod — terminate the pod and
the volume survives, then attach it to a *different* pod later. On
raceengineering, terminating deletes `/workspace` permanently, so a mistake
costs you the weights cache and any un-uploaded audio.

With a Network Volume you download the ~7 GB of weights **once, ever**.

**Cheaper compute.** Community Cloud RTX 4090 is roughly half the price of the
equivalent raceengineering instance.

**What raceengineering still wins on:** Indian regions, INR-native billing with
a GST invoice (which matters if you want input tax credit), and no FX markup.
If your accounting needs Indian invoices, that can outweigh the price gap.

---

## 1. Create a Network Volume first

Do this *before* deploying a pod, because the volume must be attached at
creation time and its region determines which GPUs you can pick.

- **Size:** 60 GB is plenty (≈7 GB weights + audio + headroom).
- **Region:** choose one that stocks RTX 4090 / A5000. `EU-RO-1` and `US-*`
  generally have the widest selection.
- **Cost:** ~$0.07/GB/month under 1 TB → **60 GB ≈ $4.20/month**, billed
  whether or not a pod is running.

That standing charge is the trade: a few dollars a month to never re-download
weights and never lose un-uploaded audio.

---

## 2. Pick the GPU

From <https://console.runpod.io/deploy>:

| GPU | VRAM | Community | Secure | Verdict |
|---|---|---|---|---|
| **RTX A5000** | 24 GB | **$0.16/hr** | $0.27/hr | **Cheapest viable.** Ampere, ample for a 1.7B model. |
| **RTX 4090** | 24 GB | **$0.34/hr** | $0.69/hr | **Recommended.** Ada, clearly faster; best price/perf. |
| L4 | 24 GB | $0.44/hr | $0.39/hr | Efficient but slower than a 4090 and costs more. |
| L40S | 48 GB | $0.79/hr | $0.99/hr | Unnecessary VRAM. |
| A100 / H100 | 80 GB | $1.19–2.69/hr | $1.39–2.99/hr | Wildly oversized for a 1.7B model. |

Qwen3-TTS-12Hz-1.7B is ~3.4 GB in bf16. Anything with 24 GB is generous.

**Community vs Secure Cloud.** Community is roughly half price but runs on
third-party hosts and can be reclaimed. For a resumable batch job that is an
acceptable trade — chunks are checkpointed, so an interrupted render resumes
where it stopped. Use Secure Cloud only if you want an uninterrupted long run.

**Launch settings**

| Setting | Value |
|---|---|
| Template | RunPod PyTorch 2.x |
| Network Volume | the one from §1, mounted at `/workspace` |
| Container disk | 20 GB (scratch only — the volume holds everything real) |

---

## 3. Run the pod FROM THE IMAGE (do not bootstrap)

The rest of this document describes bootstrapping a bare pod. **Prefer this
instead.** Two sessions were lost to hand-building the stack: the venv landed on
a volume that cannot set exec bits, torch was installed for the wrong driver,
flash-attn could not build without nvcc, and the end result was a CUDA/cuDNN
mismatch that left a 4090 at a quarter of its matmul throughput and never
finished a chapter — while reporting `cuda available: True` throughout.

Serverless never had any of those problems, because serverless runs an image.
So does this now:

1. **Build it.** Push to `feat/per-job-language` (or run the workflow by hand)
   and `.github/workflows/render-image.yml` builds `penspace/Dockerfile` on
   GitHub's amd64 runners and pushes to
   `ghcr.io/<owner>/penspace-runpods/render:latest`.
2. **Deploy a pod** with that as its **Container Image**, and attach a volume
   mounted at `/models` so the ~7GB of weights download once rather than per pod.
3. **Check the GPU is real** — 60 seconds, before you trust anything:

   ```bash
   python -m penspace.cli doctor
   ```

4. **Run the worker**:

   ```bash
   python -m penspace.worker --api https://penspace.in/api --token "$AUDIO_WORKER_TOKEN"
   ```

`penspace/Dockerfile` now shares a base image with `Dockerfile.serverless` on
purpose. Two images claiming to run the same code on different runtimes is
exactly how "it works on serverless but not on a pod" happens — change them
together or not at all.

---

## 3b (legacy). Upload and bootstrap

Add your SSH public key under **Settings → SSH Public Keys** first, then take
the connection command from the pod's **Connect** panel.

```bash
scp -P <port> -i ~/.ssh/id_ed25519 penspace-deploy.tgz root@<host>:/workspace/
ssh -p <port> -i ~/.ssh/id_ed25519 root@<host>

cd /workspace && tar xzf penspace-deploy.tgz
bash penspace/bootstrap.sh
```

`bootstrap.sh` is provider-agnostic and idempotent — it already targets
`/workspace`. It installs `ffmpeg`/`sox`, builds the venv, installs the pinned
stack, compiles FlashAttention 2 (falling back to SDPA), verifies CUDA, and
pre-downloads the weights.

**On a Network Volume this is a one-time cost.** Later pods skip straight to
§4, because the venv and `hf-cache` are already on the volume.

RunPod also offers `runpodctl` for file transfer if you prefer it to `scp`.

---

## 3b. Check the GPU is actually being used

```bash
python -m penspace.cli doctor
```

**Do this before rendering anything.** A pod once reported `cuda available:
True`, loaded the model into 3.9 GB of VRAM, and then generated at 0% GPU and
130% CPU — a hundredth of the expected speed, while billing GPU rates. Nothing
in the normal logs said so; everything looked right.

`doctor` times the generate call with CUDA events and compares that against the
wall clock. If the GPU accounts for less than half the time, it says so and
exits non-zero. It also prints the realtime factor, which is the only number
that decides whether renting a GPU beats paying per character.

---

## 4. Pin the voice

Add to `~/.bashrc` on the pod:

```bash
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf-cache
export PENSPACE_REF_AUDIO=/workspace/penspace_ref.mp3
export PENSPACE_REF_TEXT="A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting, or receive criticism you secretly feared was true. The event may last only a few minutes, but the mind continues it for hours."
```

`PENSPACE_REF_AUDIO` switches the pipeline to the Base checkpoint and clone
mode automatically.

**Pin `PENSPACE_REF_TEXT` rather than letting it auto-transcribe.** The render
id hashes the text plus every voice setting, including this transcript. Whisper
runs at a different precision on CUDA than on Apple Silicon, so an
auto-transcribed reference could differ by a word and silently produce new S3
keys for otherwise identical audio.

---

## 5. Render

```bash
# No GPU needed. Catches normalization problems for free.
python -m penspace.cli estimate --manifest manifests/rest.jsonl

# Time one book. This number sizes everything else.
time python -m penspace.cli render \
    --manifest manifests/rest.jsonl --work-dir /workspace/out

# Full catalog, straight to S3.
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=ap-south-1
python -m penspace.cli render \
    --manifest manifests/penspace.jsonl \
    --work-dir /workspace/out --bucket penspace-audio
```

Exit code `2` means some chunks never passed QA; the failing ids print to
stderr. Everything else still rendered.

**If anything tells you to `export PENSPACE_ATTN=sdpa`, do not.** That is the
fallback path, and it measured 737 seconds to produce 5.8 seconds of audio --
about 1% of normal speed. `bootstrap.sh` now installs a prebuilt flash-attn
wheel and refuses to finish without one.

CUDA settings resolve automatically: bf16, FlashAttention 2, and the batch size
un-caps from the Apple-Silicon limit of 4 back to 8.

---

## 6. Cost

Compute for ~500 titles (~167 hours of audio), assuming **10× realtime** —
a guess to be replaced with your §5 measurement:

| Option | Rate | ~17 GPU-hr | Plus storage |
|---|---|---|---|
| RTX A5000 Community | $0.16/hr | **≈ $2.70** | +$4.20/mo volume |
| RTX 4090 Community | $0.34/hr | **≈ $5.80** | +$4.20/mo volume |
| RTX 4090 Secure | $0.69/hr | ≈ $11.70 | +$4.20/mo volume |
| raceengineering RTX 4090 | ₹50/hr | ≈ ₹850 + 18% GST | included |

Compute is close to free at this scale. **The volume's standing monthly charge
will likely exceed your compute bill** — which is fine, but it means: delete
the Network Volume if you go months without rendering.

Stopped pods still bill for container disk, so **terminate** the pod between
batches rather than stopping it. The Network Volume is what makes that safe.

---

## 7. Operating loop

```
create volume (once) → deploy pod with volume → bootstrap (first pod only)
   → estimate → render → verify QA → upload to S3 → TERMINATE the pod
```

Terminating is safe here — the volume keeps the venv, weights and outputs. Next
time, deploy a new pod against the same volume and go straight to rendering.

Renders are resumable at chunk granularity: completed chunks are cached under
`/workspace/out/<id>/<render_id>/chunks/`, so re-running a manifest
re-synthesizes only what is missing and skips summaries already uploaded. A
reclaimed Community Cloud pod costs you one batch, not the run.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `flash-attn` build fails | Harmless. `export PENSPACE_ATTN=sdpa`. |
| CUDA OOM | Lower `--batch-size` (default 8). |
| Chunks hit the generation ceiling | Lower `--max-chunk-chars`; the model caps at ~164 s of audio per call. |
| Exit code 2 | Chunks failed QA after retries. Inspect the printed ids; raise `--max-wer` only if the audio actually sounds fine. |
| Re-downloading weights each pod | `HF_HOME` not on the Network Volume, or no volume attached. |
| Same audio, new S3 keys | `PENSPACE_REF_TEXT` not pinned — see §4. |
| No GPUs available in region | Your volume's region has none free. Volumes are region-locked; you may need one in a second region. |
| Pod vanished mid-render | Community Cloud reclaim. Re-deploy and re-run — cached chunks resume. |

---

## 9. After rendering

`--bucket` writes:

```
{prefix}/{summary_id}/{render_id}/audio.m4a
{prefix}/{summary_id}/{render_id}/timing.json
```

plus `/workspace/out/catalog.jsonl` — one row per playable summary with keys and
duration. That file is the handoff to the app.

Keep the bucket private and mint presigned URLs behind your entitlement check;
`penspace/serve_urls.py` is a reference implementation.

---

## A note on Serverless

RunPod Serverless scales to zero and bills per second, which sounds ideal — but
it is priced well above an equivalent on-demand pod per active hour, so it only
wins when a box would otherwise sit idle awaiting sporadic requests.

Pre-generating a catalog is the opposite: dense, predictable, non-interactive
work. A pod is both cheaper and simpler. Serverless would only make sense if
Penspace later needs on-demand synthesis for user-generated summaries.
