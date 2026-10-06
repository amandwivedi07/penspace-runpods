# The render image for a GPU POD.
#
#   docker build -t <user>/penspace-render:latest -f penspace/Dockerfile .
#   docker push  <user>/penspace-render:latest
#
# Then deploy a RunPod pod with that as its container image and run the worker
# inside it. Do NOT bootstrap a bare pod and install this stack by hand: two
# sessions were lost that way. Building torch wheels over a provider's base
# image produced a CUDA/cuDNN mismatch that left a 4090 running at a quarter of
# its matmul throughput and never finished a single chapter, while reporting
# `cuda available: True` the whole time. The image is the fix.
#
#   docker run --gpus all --rm \
#     -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
#     -e PENSPACE_S3_BUCKET -e PENSPACE_REF_AUDIO -e PENSPACE_REF_TEXT \
#     -v /workspace/models:/models -v /workspace/out:/work \
#     <user>/penspace-render:latest \
#     python -m penspace.worker --api https://penspace.in/api --token "$AUDIO_WORKER_TOKEN"

# THE SAME BASE AS Dockerfile.serverless, DELIBERATELY.
#
# This file used to pin pytorch 2.5.1 / cuda 12.4 while the serverless image —
# the only environment that has ever rendered audio at a usable speed — was on
# 2.8.0 / cuda 12.8. Two images claiming to run the same code on two different
# runtimes is how "it works on serverless but not on a pod" happens. Change
# them together or not at all.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# HF_HOME is where the model cache lands. Point it at a mounted volume
# (-v /workspace/models:/models) so a replacement pod does not re-download ~7GB
# of weights while the GPU sits idle and billed.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PENSPACE_WORK_DIR=/work \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY penspace/requirements.txt /app/penspace/requirements.txt
RUN pip install --no-cache-dir -r penspace/requirements.txt

# FlashAttention 2 is attempted but NOT required.
#
# `resolve_attn` asks for flash_attention_2 on any CUDA device, so a failed
# build would make the model load raise at runtime rather than degrade. Pinning
# sdpa below removes that trap. It is also what the working serverless image
# does — so sdpa is the PROVEN path here, not a consolation prize.
RUN MAX_JOBS=4 pip install --no-cache-dir -U flash-attn --no-build-isolation || \
    echo "flash-attn build failed — sdpa is pinned below, so this is survivable"
ENV PENSPACE_ATTN=sdpa

# The Qwen weights are NOT baked in.
#
# They were once, and it made the image 13.85 GB. A pod pulls the whole image
# before it runs anything, and at that size pods were slow to become usable.
# Mount a volume at /models instead; the first run populates it and every pod
# after that starts immediately.

COPY penspace /app/penspace
COPY penspace_ref.mp3 /app/penspace_ref.mp3

# The second reference take, slate removed.
#
# penspace_ref.mp3 predates the recording the narrator actually supplied: same
# script, a different take, and nobody could say from the waveform whether it
# was even the same person — correlation between the two is 0.02, and pitch
# alone cannot tell two similar voices apart. julian_ref.mp3 is the supplied
# file, and it opens with "One play." — a take slate that would be cloned along
# with the voice. This is that file from 0.82s, where the slate ends.
#
# Its transcript is identical to penspace_ref.mp3's, so PENSPACE_REF_TEXT does
# not change when switching between them. Point PENSPACE_REF_AUDIO at
# /app/julian_ref_clean.mp3 to use it. Both ship so a render can be compared
# against either, and so switching back costs an env var rather than a rebuild.
COPY julian_ref_clean.mp3 /app/julian_ref_clean.mp3

# Two references cut from a rendered narration, not from a recording of anyone.
#
# These are OUTPUT: audio this pipeline produced earlier. Cloning from them
# clones a clone, and the drift is measurable — a clip taken from the long one
# has a pitch spread of 38.6 Hz against the 32.6 Hz of the recording it
# descends from, and 69.5% voiced frames against 59.8%. Each generation
# compounds that. They ship because they were asked for and because judging
# them needs a render, not an argument; the transcripts beside them in refs/
# are what PENSPACE_REF_TEXT must be set to, exactly.
#
#   /app/hf_ref_30s.mp3   31.8s, cut on a sentence boundary  -> refs/hf_ref_30s.txt
#   /app/hf_ref_full.mp3  166.3s, the whole file             -> refs/hf_ref_full.txt
#
# Prefer the 30s one. Cloning wants roughly 10-30 seconds of clean speech;
# past that it adds little and the prompt gets expensive to build.
COPY hf_ref_30s.mp3 /app/hf_ref_30s.mp3
COPY hf_ref_full.mp3 /app/hf_ref_full.mp3
COPY refs /app/refs

# PENSPACE_WORK_DIR points here, and nothing else creates it. Without this the
# worker dies on its first chapter with "No such file or directory" — which it
# did, on a pod that was otherwise correct.
RUN mkdir -p /work

# No ENTRYPOINT on purpose. A pod runs either the pull worker or the batch CLI,
# and which one is a runtime decision:
#
#   python -m penspace.worker --api ... --token ...   # pull jobs from the panel
#   python -m penspace.cli doctor                     # 60s: is the GPU real?
#   python -m penspace.cli render --manifest ...      # render a manifest
#
# Run `doctor` first on any new pod. It reports the realtime factor, which is
# the only number that says whether renting this GPU beats paying per character.
CMD ["python", "-m", "penspace.cli", "doctor"]
