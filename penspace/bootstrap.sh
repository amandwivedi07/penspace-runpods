#!/usr/bin/env bash
# Prepare a fresh GPU pod for Penspace catalog rendering.
#
# Written for raceengineering.ai pods, but nothing here is provider-specific
# beyond the /workspace convention.
#
#   bash penspace/bootstrap.sh
#
# Everything that is expensive to recreate -- the venv, the model weights, the
# render outputs -- lives under /workspace, which is the only directory that
# survives a pause/resume. Terminating the pod deletes it, so push finished
# audio to S3 before you terminate.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/venv"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ ! -d "$WORKSPACE" ]; then
    echo "No $WORKSPACE directory. Set WORKSPACE=... if your pod differs." >&2
    exit 1
fi

log "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "nvidia-smi failed -- is this actually a GPU pod?" >&2
    exit 1
}

log "System packages"
if command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # ffmpeg encodes the m4a; qwen-tts shells out to the sox binary.
    apt-get install -y -qq ffmpeg sox libsndfile1 git curl >/dev/null
fi
ffmpeg -version | head -1
sox --version | head -1

log "Python environment at $VENV"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip wheel

log "Dependencies"
pip install -q -r "$(dirname "$0")/requirements.txt"
pip install -q python-docx  # for the ingest command

log "PyTorch / CUDA"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

log "FlashAttention 2 (optional, CUDA only)"
# Builds from source and is slow. If it fails the pipeline falls back to SDPA,
# which is correct but uses more memory.
if python -c "import flash_attn" 2>/dev/null; then
    echo "already installed"
elif MAX_JOBS=4 pip install -q -U flash-attn --no-build-isolation 2>/dev/null; then
    echo "installed"
else
    echo "build failed -- export PENSPACE_ATTN=sdpa before rendering"
fi

log "Pre-downloading model weights to $HF_HOME"
# Do this once, on the persistent volume. Otherwise every resume re-downloads
# ~7GB while the GPU sits idle and billed.
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in (
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
):
    print("fetching", repo, flush=True)
    snapshot_download(repo)
print("weights ready")
PY

log "Done"
cat <<EOF

Environment is ready. Add these to your shell (or ~/.bashrc):

    source $VENV/bin/activate
    export HF_HOME=$HF_HOME

Then verify with a short render:

    python -m penspace.cli audition --speakers Ryan Aiden --out $WORKSPACE/audition

Reminders:
  * Only $WORKSPACE survives pause/resume. Terminate deletes it.
  * Upload finished audio to S3 BEFORE terminating the pod.
  * Billing is per-minute -- pause the pod the moment a batch finishes.
EOF
