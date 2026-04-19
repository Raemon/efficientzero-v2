#!/usr/bin/env bash
# One-shot setup on a fresh RunPod (or similar) GPU box.
# Assumes the base image already has CUDA + a recent PyTorch.
#
# Usage on the pod:
#   git clone https://github.com/<you>/efficientzero-v2.git
#   cd efficientzero-v2
#   bash scripts/setup_remote.sh

set -euo pipefail

echo "[setup] python: $(python --version)"
echo "[setup] gpu:"
nvidia-smi -L || echo "  (no nvidia-smi found — are you on a CPU box?)"

echo "[setup] installing package + extras"
pip install --upgrade pip
pip install -e ".[atari,dmc,dev]"

echo "[setup] verifying torch sees the GPU"
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[setup] WANDB_API_KEY found in env, logging in"
  wandb login --relogin "$WANDB_API_KEY"
else
  echo "[setup] no WANDB_API_KEY in env. Run: wandb login   (or export WANDB_API_KEY=...)"
fi

echo "[setup] running smoke test (CPU is fine here)"
python scripts/train.py --config configs/smoke.yaml trainer.total_train_steps=100 logging.wandb=false

echo "[setup] done. To start a real run inside tmux:"
echo "  tmux new -s train"
echo "  make train-atari       # or train-dmc-proprio / train-dmc-vision"
echo "  (Ctrl-b d to detach)"
