#!/usr/bin/env bash
# Launch the Penspace browser studio on the GPU pod.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/venv"

source "$VENV/bin/activate"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export PENSPACE_ATTN="${PENSPACE_ATTN:-sdpa}"
export PENSPACE_DEVICE="${PENSPACE_DEVICE:-cuda:0}"
export PENSPACE_WORKSPACE="$WORKSPACE"
export PENSPACE_MANIFEST_DIR="$WORKSPACE/manifests"
export PENSPACE_WORK_DIR="$WORKSPACE/out"

# Voice clone settings (edit if your reference recording differs).
export PENSPACE_REF_AUDIO="${PENSPACE_REF_AUDIO:-$WORKSPACE/penspace_ref.mp3}"
export PENSPACE_REF_TEXT="${PENSPACE_REF_TEXT:-A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting, or receive criticism you secretly feared was true. The event may last only a few minutes, but the mind continues it for hours.}"

if ! python -c "import fastapi, uvicorn, multipart, docx" 2>/dev/null; then
    pip install -q fastapi "uvicorn[standard]" python-multipart python-docx
fi

cd "$WORKSPACE"
echo ""
echo "Penspace Studio starting on http://0.0.0.0:8080"
echo "Open the pod Web URL, or from your Mac:"
echo "  ssh -L 8080:localhost:8080 -p <port> root@<host>"
echo "  open http://localhost:8080"
echo ""

exec uvicorn penspace.studio:app --host 0.0.0.0 --port 8080
