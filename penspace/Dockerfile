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
# The reference recording for voice cloning, baked in so the voice cannot drift
# between pods and so a pod needs no volume mounted just to hold one small file.
#
# This is the narrator's own take with its "One play." slate trimmed off at
# 0.82s. It replaced penspace_ref.mp3, a different take of the same script that
# had been committed two months earlier and that every chapter up to this point
# was cloned from. Nothing in the audio could settle whether the two were even
# the same person — correlation between them was 0.02, and pitch and spread
# cannot tell two similar male voices apart — so the question was settled by
# listening, and the older file is gone to stop the two being confused again.
#
# Chapters rendered before this keep their audio in S3; only a re-render picks
# up the new voice, because the reference is part of the render fingerprint.
COPY julian_ref_clean.mp3 /app/julian_ref_clean.mp3

# Julian, as rendered by Cartesia (Sonic 3.6) — the voice chosen after
# listening, used as the cloning reference.
#
# This is SYNTHETIC: Cartesia's own clone of the narrator, so the renderer
# imitates that rendition rather than the recording behind it. It was chosen
# for how it sounds; whether a clone of a clone holds up across a catalogue is
# what the first renders are for.
#
# Cut to the whole first paragraph of the Who Will Cry When You Die?
# introduction — 2.844s to 28.579s, 25.7s, at the quietest frame in each gap,
# with 20ms fades. The title line before it is left out: a reference teaches
# the voice, and a book title read as a header would be learned as part of it.
#
# PENSPACE_REF_TEXT for this clip is julian_ref_cartesia.txt, word for word,
# and it differs from julian_ref_clean.mp3's. Switch both together.
COPY julian_ref_cartesia.mp3 /app/julian_ref_cartesia.mp3
COPY julian_ref_cartesia.txt /app/julian_ref_cartesia.txt

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
