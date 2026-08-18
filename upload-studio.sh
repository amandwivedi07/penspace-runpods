#!/usr/bin/env bash
# Upload studio files to the GPU pod and start the web UI.
#
# Usage:
#   ./upload-studio.sh 'ssh root@69.162.125.131 -p 32771 -i ~/.ssh/id_ed25519'
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 '<ssh command from Endpoints panel>'" >&2
    exit 1
fi

SSH_CMD="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

PORT=""
HOST=""
read -r -a PARTS <<< "$SSH_CMD"
for i in "${!PARTS[@]}"; do
    case "${PARTS[$i]}" in
        -p) PORT="${PARTS[$((i + 1))]}" ;;
        root@*) HOST="${PARTS[$i]#root@}" ;;
    esac
done

if [ -z "$PORT" ] || [ -z "$HOST" ]; then
    echo "Could not parse SSH command: $SSH_CMD" >&2
    exit 1
fi

echo "==> Uploading studio files"
scp -P "$PORT" -i "$SSH_KEY" \
    "$SCRIPT_DIR/penspace/studio.py" \
    "$SCRIPT_DIR/penspace/studio.html" \
    "$SCRIPT_DIR/penspace/start_studio.sh" \
    "root@$HOST:/workspace/penspace/"

echo "==> Starting Penspace Studio on port 8080"
ssh -p "$PORT" -i "$SSH_KEY" "root@$HOST" <<'REMOTE'
chmod +x /workspace/penspace/start_studio.sh
# Kill any previous studio instance on 8080
pkill -f "uvicorn penspace.studio" 2>/dev/null || true
nohup bash /workspace/penspace/start_studio.sh > /workspace/studio.log 2>&1 &
sleep 2
tail -5 /workspace/studio.log || true
REMOTE

echo ""
echo "Studio is running. Share with your manager:"
echo "  1. Click 'Web URL' on the pod dashboard (easiest), OR"
echo "  2. Tunnel from your Mac and send http://localhost:8080:"
echo "       ssh -L 8080:localhost:8080 -p $PORT -i $SSH_KEY root@$HOST"
echo ""
echo "Manager guide: MANAGER_STUDIO_GUIDE.txt"
