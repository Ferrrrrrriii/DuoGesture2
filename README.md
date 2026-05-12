# DuoGesture

**Co-speech gesture generation** — holistic full-body motion from speech audio.

> Paper: [SemTalk: Holistic Co-speech Motion Generation with Semantic Grounding](https://arxiv.org/abs/2412.16563) (ICCV 2025)  
> Project page: [https://duogesture.github.io/DuoGesture/](https://duogesture.github.io/DuoGesture/)

---

## Prerequisites

| Requirement | How to get it |
|---|---|
| Python 3.8+, CUDA GPU | Required for training (4× GPU recommended) |
| BEAT2 dataset | See step 3 in setup below |
| Model weights | See step 4 in setup below |
| Preprocessed dataset cache | Build after downloading BEAT2 |

---

## Setup (one-time)

```bash
# Clone and enter the repo
git clone https://github.com/Ferrrrrrriii/DuoGesture2.git
cd DuoGesture2

# Run the one-shot setup (installs deps, downloads BEAT2, weights, HuBERT, Whisper)
bash setup.sh
```

After `setup.sh` completes, build the dataset cache:

```bash
python dataloaders/save_train_dataset.py --config configs/duogesture_moclip_sparse.yaml
python dataloaders/save_test_dataset.py  --config configs/duogesture_moclip_sparse.yaml
```

---

## Required directory layout

```
DuoGesture2/
├── BEAT2/
│   └── beat_english_v2.0.0/   ← downloaded by setup.sh
├── datasets/
│   ├── beat2_duogesture_train_moclip/   ← built by save_train_dataset.py
│   └── beat2_duogesture_test_moclip.pkl ← built by save_test_dataset.py
├── facebook/
│   └── hubert-large-ls960-ft/  ← downloaded by setup.sh
├── Systran/
│   └── faster-whisper-large-v3/ ← downloaded by setup.sh
└── weights/
    ├── AESKConv_240_100.bin         ← RVQ-VAE eval model
    ├── best_gate_abl_A_fgd0406.bin  ← best checkpoint (FGD 0.4056)
    ├── pretrained_vq/               ← codebook weights
    └── moclip_checkpoints/          ← TMR text encoder
```

---

## Training

```bash
# Stage 1 — base motion (single GPU)
python train.py --config configs/duogesture_base.yaml

# Stage 2 — sparse semantic (single GPU)
python train.py --config configs/duogesture_sparse.yaml

# Stage 2 — MoCLIP sparse, best result (4 GPUs via torchrun)
torchrun --nproc_per_node=4 train_torchrun.py --config configs/duogesture_moclip_sparse.yaml

# Or single GPU:
python train.py --config configs/duogesture_moclip_sparse.yaml
```

---

## Evaluation

```bash
# Test state on best checkpoint
python train.py --test_state \
  --config configs/duogesture_moclip_sparse.yaml \
  --load_ckpt weights/best_gate_abl_A_fgd0406.bin

# FGD metric
python utils/run_fgd_eval.py \
  --checkpoint weights/best_gate_abl_A_fgd0406.bin \
  --config configs/duogesture_moclip_sparse.yaml
```

Best known result: **FGD = 0.4056** (epoch 242, MoCLIP sparse, no VIB, no physics).

---

## GPU configuration

Edit `gpus:` in the config to match your setup:

```yaml
# Single GPU
gpus: [0]
ddp: False

# 4 GPUs
gpus: [0,1,2,3]
ddp: True
```
