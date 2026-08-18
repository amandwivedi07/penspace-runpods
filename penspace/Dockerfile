# Render image for the Penspace narration batch job.
#
# Build once, push to a registry, then pull on the GPU VM. Baking the model
# weights in (see the optional stage below) means a fresh VM starts rendering
# immediately instead of burning paid GPU minutes on a Hugging Face download.
#
#   docker build -t penspace-render -f penspace/Dockerfile .
#   docker run --gpus all --rm \
#     -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
#     -v $PWD/work:/work -v $PWD/manifests:/manifests \
#     penspace-render render --manifest /manifests/catalog.jsonl \
#       --work-dir /work --bucket penspace-audio

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY penspace/requirements.txt /app/penspace/requirements.txt
RUN pip install --no-cache-dir -r penspace/requirements.txt

# FlashAttention 2 cuts VRAM and speeds up generation. It builds from source and
# is slow; if it fails on your GPU, drop it and pass --device with
# PENSPACE_ATTN=sdpa instead.
RUN MAX_JOBS=4 pip install --no-cache-dir -U flash-attn --no-build-isolation || \
    echo "flash-attn build failed; set PENSPACE_ATTN=sdpa at runtime"

COPY penspace /app/penspace

# Optional: bake weights into the image so VM boot does not pay for the download.
# RUN pip install --no-cache-dir -U "huggingface_hub[cli]" && \
#     hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice && \
#     hf download Qwen/Qwen3-TTS-Tokenizer-12Hz

ENTRYPOINT ["python", "-m", "penspace.cli"]
CMD ["--help"]
