#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/duogesture_moclip_sparse.yaml"
EPOCHS="${1:-200}"

cd "$REPO_ROOT"

echo "Starting DuoGesture retraining with config: $CONFIG"
echo "Epochs: $EPOCHS"
python train.py --config "$CONFIG" --epochs "$EPOCHS"
