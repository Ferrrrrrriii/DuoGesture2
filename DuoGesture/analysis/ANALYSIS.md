# DuoGesture Analysis Summary

## Best FGD result

- Best model checkpoint: `weights/best_gate_abl_A_fgd0406.bin`
- Best FGD score: `0.4056`
- Config used: `configs/duogesture_moclip_sparse.yaml`

## Recommended workflow

1. Install dependencies from `requirements_fixed.txt`
2. Prepare BEAT2 dataset under `BEAT2/beat_english_v2.0.0/`
3. Run the best checkpoint test:
   - `bash DuoGesture/scripts/run_best_fgd.sh`
4. Retrain the full model or fine-tune from the selected checkpoint:
   - `bash DuoGesture/scripts/retrain.sh 200`

## What this folder captures

- `configs/duogesture_moclip_sparse.yaml`: best known DuoGesture MoCLIP config
- `configs/duogesture_base.yaml`: base motion training config
- `configs/duogesture_sparse.yaml`: sparse semantic training config
- `scripts/run_best_fgd.sh`: wrapper for quick best checkpoint evaluation
- `scripts/retrain.sh`: wrapper for retraining with a configurable epoch count

## Notes for reviewers

- This folder is intended to make DuoGesture easier to inspect and run for evaluators.
- The best checkpoint is not stored here, but the wrapper points to the canonical best-FGD file in `weights/`.
- If the repository is copied to a new location, keep `DuoGesture/` next to the code root so the wrappers still work.
