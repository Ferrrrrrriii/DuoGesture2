#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="$REPO_ROOT/weights/best_gate_abl_A_fgd0406.bin"
CONFIG="$REPO_ROOT/configs/duogesture_moclip_sparse.yaml"

cd "$REPO_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  echo "Place the best-FGD checkpoint into the repository weights/ directory." >&2
  exit 1
fi

echo "Running DuoGesture best-FGD test state with checkpoint: $CHECKPOINT"
python train.py --test_state --config "$CONFIG" --load_ckpt "$CHECKPOINT"

echo "Running FGD evaluation on the best checkpoint"
python utils/run_fgd_eval.py --checkpoint "$CHECKPOINT" --config "$CONFIG"
