#!/usr/bin/env bash
# One-shot pod preparation for Penspace narration.
#
# Everything a fresh pod needs, in the order that works, with the two traps
# that cost hours already baked in:
#
#   - the venv goes on CONTAINER DISK, not the network volume: python -m venv
#     cannot set exec bits there and ensurepip dies.
#   - torch is pinned to the cu128 build. requirements.txt pins no torch at
#     all, so pip takes whatever CUDA build PyPI is serving and the A40's
#     driver (12.8) cannot run a cu130 wheel.
#   - the HF cache goes on container disk too. The global volume corrupted
#     both model.bin and config.json; re-downloading 5 GB beats debugging it.
set -euo pipefail
log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

WORKSPACE=/root/pens
export HF_HOME=$WORKSPACE/hf-cache

log "Unpacking"
mkdir -p "$WORKSPACE"
cd /workspace && tar xzf penspace-deploy.tgz 2>/dev/null || true
rm -f /workspace/._* /workspace/penspace/._* 2>/dev/null || true
ls -lh /workspace/andre_ref.mp3

log "Bootstrap (venv on container disk)"
WORKSPACE=$WORKSPACE HF_HOME=$HF_HOME bash /workspace/penspace/bootstrap.sh || true

log "Pinning torch to the cu128 build"
"$WORKSPACE/venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
"$WORKSPACE/venv/bin/pip" install -q hf_transfer
"$WORKSPACE/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

log "Fetching the voice model"
"$WORKSPACE/venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-TTS-Tokenizer-12Hz", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
    print("fetching", repo, flush=True); snapshot_download(repo)
print("weights ready")
PY

log "Ready"
cat <<'MSG'
Now, in an interactive session:

  set -a; . /root/pens/.secrets; set +a
  source /root/pens/venv/bin/activate
  export HF_HOME=/root/pens/hf-cache PENSPACE_S3_BUCKET=penspace-audio AWS_REGION=ap-south-1 \
         PENSPACE_WHISPER_MODEL=small PENSPACE_REF_AUDIO=/workspace/andre_ref.mp3 PENSPACE_QA=false \
         PENSPACE_REF_TEXT="A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting,  or receive criticism you secretly feared was true. The event may last only a few minutes,  but the mind continues it for hours."
  cd /workspace && python -m penspace.worker --api <TUNNEL>/api --token $AUDIO_WORKER_TOKEN --batch 2
MSG
