#!/usr/bin/env bash
# DuoGesture — one-shot setup script
# Downloads all required weights and preprocessed data caches.
# Run once before training or evaluation.
set -euo pipefail

# Always run from the directory containing this script
cd "$(dirname "$(realpath "$0")")"

# Prefer conda env's Python and git over stale system binaries
# $CONDA_PREFIX is set automatically by `conda activate`; fall back to detection
CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
elif [[ -n "$CONDA_BASE" ]]; then
  for candidate in "$CONDA_BASE/envs/gesturelsm/bin" "$CONDA_BASE/envs/semtalk/bin" "$CONDA_BASE/bin"; do
    if [[ -x "$candidate/python" ]]; then
      export PATH="$candidate:$PATH"
      break
    fi
  done
fi

PYTHON=$(command -v python3 || command -v python)
echo "Using Python: $PYTHON  ($(${PYTHON} --version 2>&1))"

# Point pkg-config at conda env's ffmpeg so av (PyAV) can build
CONDA_PKG_CONFIG="${CONDA_PREFIX:-}/lib/pkgconfig"
if [[ -d "$CONDA_PKG_CONFIG" ]]; then
  export PKG_CONFIG_PATH="$CONDA_PKG_CONFIG:${PKG_CONFIG_PATH:-}"
fi

echo "=== DuoGesture Setup ==="

# 1. Python dependencies
echo "[1/4] Installing Python dependencies..."
"$PYTHON" -m pip install -r requirements.txt

# 2. HuBERT + Whisper (needed by dataloader)
echo "[2/4] Downloading HuBERT and Whisper models..."
"$PYTHON" - << 'PYEOF'
from huggingface_hub import snapshot_download
import os
os.makedirs('facebook/hubert-large-ls960-ft', exist_ok=True)
os.makedirs('Systran/faster-whisper-large-v3', exist_ok=True)
print("  Downloading HuBERT (~1.3 GB)...")
snapshot_download(repo_id="facebook/hubert-large-ls960-ft",
                  local_dir="facebook/hubert-large-ls960-ft", resume_download=True)
print("  Downloading Whisper large-v3 (~3 GB)...")
snapshot_download(repo_id="Systran/faster-whisper-large-v3",
                  local_dir="Systran/faster-whisper-large-v3", resume_download=True)
PYEOF

# 3. BEAT2 dataset
echo "[3/4] Downloading BEAT2 dataset (English subset)..."
"$PYTHON" - << 'PYEOF'
from huggingface_hub import snapshot_download
import os
os.makedirs('BEAT2', exist_ok=True)
print("  Downloading BEAT2 (~several GB, may take a while)...")
snapshot_download(repo_id="H-Liu1997/BEAT2",
                  local_dir="BEAT2", repo_type="dataset", resume_download=True)
PYEOF

# 4. DuoGesture pretrained weights
echo "[4/4] Downloading DuoGesture pretrained weights..."
"$PYTHON" - << 'PYEOF'
from huggingface_hub import hf_hub_download, snapshot_download
import os

os.makedirs('weights/pretrained_vq', exist_ok=True)
os.makedirs('weights/moclip_checkpoints/models', exist_ok=True)

# RVQ-VAE codebook weights (required by both trainers)
VQ_FILES = [
    "AESKConv_240_100.bin",
    "pretrained_vq/rvq_face_600.bin",
    "pretrained_vq/rvq_hands_500.bin",
    "pretrained_vq/rvq_upper_500.bin",
    "pretrained_vq/rvq_lower_600.bin",
    "pretrained_vq/last_1700_foot.bin",
]
for f in VQ_FILES:
    dest = os.path.join("weights", f)
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        continue
    print(f"  Downloading {f}...")
    hf_hub_download(repo_id="DuoGesture/DuoGesture-weights", filename=f,
                    local_dir="weights")

# MoCLIP / TMR text encoder
print("  Downloading TMR text encoder...")
snapshot_download(
    repo_id="DuoGesture/DuoGesture-weights",
    local_dir="weights/moclip_checkpoints",
    allow_patterns=["models/tmr_humanml3d_guoh3dfeats/**"],
    resume_download=True)

print("  All weights downloaded.")
PYEOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Place your best checkpoint in weights/ (e.g. best_gate_abl_A_fgd0406.bin)"
echo "  2. Build the dataset cache:"
echo "       python dataloaders/save_train_dataset.py --config configs/duogesture_moclip_sparse.yaml"
echo "       python dataloaders/save_test_dataset.py  --config configs/duogesture_moclip_sparse.yaml"
echo "  3. Train (4 GPUs):"
echo "       python train_torchrun.py --config configs/duogesture_moclip_sparse.yaml"
echo "  3b. Train (1 GPU):"
echo "       python train.py --config configs/duogesture_moclip_sparse.yaml"
