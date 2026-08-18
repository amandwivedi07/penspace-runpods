#!/usr/bin/env bash
# Upload penspace-deploy.tgz to a raceengineering.ai pod and bootstrap.
#
# Usage:
#   ./deploy-to-pod.sh '<ssh command from Endpoints panel>'
#
# Example:
#   ./deploy-to-pod.sh 'ssh -p 12345 root@1.2.3.4'
#
# The SSH command is shown on your pod card under Endpoints. Copy it exactly —
# the port changes every time you pause/resume.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 '<ssh command from Endpoints panel>'" >&2
    echo "Example: $0 'ssh -p 12345 root@1.2.3.4'" >&2
    exit 1
fi

SSH_CMD="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE="$SCRIPT_DIR/penspace-deploy.tgz"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [ ! -f "$BUNDLE" ]; then
    echo "Missing $BUNDLE — run from Qwen3-TTS root or recreate the bundle." >&2
    exit 1
fi

# Parse host and port from the SSH command.
# Supports: ssh -p PORT root@HOST  or  ssh root@HOST -p PORT
PORT=""
HOST=""
read -r -a PARTS <<< "$SSH_CMD"
for i in "${!PARTS[@]}"; do
    case "${PARTS[$i]}" in
        -p)
            PORT="${PARTS[$((i + 1))]}"
            ;;
        root@*)
            HOST="${PARTS[$i]#root@}"
            ;;
    esac
done

if [ -z "$PORT" ] || [ -z "$HOST" ]; then
    echo "Could not parse host/port from: $SSH_CMD" >&2
    echo "Expected something like: ssh -p 12345 root@1.2.3.4" >&2
    exit 1
fi

echo "==> Uploading bundle to root@$HOST:$PORT:/workspace/"
scp -P "$PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    "$BUNDLE" "root@$HOST:/workspace/"

echo "==> Extracting and bootstrapping on pod (≈10 min first time)..."
ssh -p "$PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "root@$HOST" <<'REMOTE'
set -euo pipefail
cd /workspace
tar xzf penspace-deploy.tgz

# Pin voice env for every shell session.
if ! grep -q 'PENSPACE_REF_AUDIO' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'BASHRC'

# Penspace narrator voice (pinned — do not auto-transcribe on CUDA)
source /workspace/venv/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/hf-cache
export PENSPACE_REF_AUDIO=/workspace/penspace_ref.mp3
export PENSPACE_REF_TEXT="A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting, or receive criticism you secretly feared was true. The event may last only a few minutes, but the mind continues it for hours."
BASHRC
fi

bash penspace/bootstrap.sh
REMOTE

echo ""
echo "==> Deploy complete."
echo ""
echo "SSH in:"
echo "  ssh -p $PORT -i $SSH_KEY root@$HOST"
echo ""
echo "Then dry-run + test render:"
echo "  source /workspace/venv/bin/activate"
echo "  export HF_HOME=/workspace/hf-cache"
echo "  export PENSPACE_REF_AUDIO=/workspace/penspace_ref.mp3"
echo "  export PENSPACE_REF_TEXT=\"A mistake happens at work. Perhaps you forget an important detail, say the wrong thing in a meeting, or receive criticism you secretly feared was true. The event may last only a few minutes, but the mind continues it for hours.\""
echo "  cd /workspace"
echo "  python -m penspace.cli estimate --manifest manifests/intro.jsonl"
echo "  time python -m penspace.cli render --manifest manifests/intro.jsonl --work-dir /workspace/out"
