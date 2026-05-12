# DuoGesture

This folder collects the key DuoGesture files, scripts, and analysis for a runnable GitHub-style project snapshot.

> Note: this folder is designed to live inside the repository root. The scripts use the parent repo root to invoke the training code, dataset, and weights.

## What is included

- `configs/` – core DuoGesture configs for base, sparse, and MoCLIP sparse training/evaluation
- `scripts/` – easy entrypoints for best-FGD evaluation and retraining
- `analysis/` – short summary of the best model, FGD results, and recommended workflow
- `.gitignore` – ignores generated outputs and checkpoints

## Quick start

From the repository root:

```bash
# Activate your Python environment
source .venv/bin/activate
# or: conda activate <your-env>

# Install dependencies from the repo root
pip install -r requirements_fixed.txt

# Run the best FGD model test + FGD evaluation
bash DuoGesture/scripts/run_best_fgd.sh

# Start retraining with the default 200 epochs
bash DuoGesture/scripts/retrain.sh 200
```

## Best model

The best known FGD checkpoint is:

- `weights/best_gate_abl_A_fgd0406.bin`

This is the checkpoint used by `DuoGesture/scripts/run_best_fgd.sh`.

## Recommendations

- Place the BEAT2 dataset under `BEAT2/beat_english_v2.0.0/`
- Use `configs/duogesture_moclip_sparse.yaml` for the best known DuoGesture result
- Use `configs/duogesture_base.yaml` and `configs/duogesture_sparse.yaml` for base and sparse training workflows
- Run `python utils/run_fgd_eval.py --checkpoint weights/best_gate_abl_A_fgd0406.bin --config configs/duogesture_moclip_sparse.yaml` to verify FGD

## Notes

- `DuoGesture/` is intentionally lightweight. The core model code remains in the root repository.
- We do not include large checkpoint files in this folder, only the path and the evaluation wrapper.
