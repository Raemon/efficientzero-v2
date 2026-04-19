# EfficientZero V2 (reference scaffold)

Reference implementation of **EfficientZero V2** (Wang et al., ICML 2024,
[arXiv:2403.00564](https://arxiv.org/abs/2403.00564)) — the latest in the
MuZero / EfficientZero line of model-based RL with MCTS planning over a
learned latent dynamics model.

This repo is structured for the standard "edit locally, train on a rented
cloud GPU" workflow.

## What's actually implemented

- **Working end to end**: networks (representation `H`, dynamics `G`,
  prediction `F`), sampled Gumbel search, prioritized replay, mixed
  value target / SVE, self-play, the four losses (reward, value, policy,
  consistency), trainer loop.
- **Smoke test**: `make smoke` trains on `CartPole-v1` end to end on CPU
  in ~1 minute. Use this to verify the pipeline before paying for a GPU.
- **Env wrappers**: Atari, DM-Control. Will run on a GPU box but are
  *not* tuned to reproduce paper numbers — that's a separate effort.
- **Not** included: distributed/multi-GPU, Docker, full benchmark
  harness across all 26 Atari games × N seeds.

## Workflow

```
[Your Mac]                          [RunPod GPU box]
edit code                           git pull
git push       ─── GitHub ─────►    python scripts/train.py
                                    │
W&B dashboard  ◄─── wandb.ai ───────┘
```

You only pay while a GPU is actively running. Stop the pod when done.

## Quickstart (local, CPU)

```bash
pip install -e ".[dev]"
make smoke              # trains on CartPole, ~60s on CPU
make test               # runs unit tests
```

If `make smoke` works, the pipeline is wired up correctly.

## Quickstart (cloud GPU)

See [`scripts/RUNPOD.md`](scripts/RUNPOD.md) for the full step-by-step.
TL;DR:

1. Launch a RunPod **PyTorch 2.x** template on a single RTX 3090.
2. SSH in. Run `bash scripts/setup_remote.sh` (clones repo, installs deps, logs into W&B).
3. `tmux new -s train` then `make train-atari` (or whichever config).
4. Detach with `Ctrl-b d`, watch metrics on https://wandb.ai.
5. When done: `runpodctl stop pod` (or stop in the web UI). Don't forget.

## Repository layout

```
configs/                 YAML hyperparam files, one per environment
scripts/
  train.py               Entry point (argparse + config loading)
  eval.py                Load checkpoint, run eval episodes
  setup_remote.sh        One-shot setup on a fresh RunPod box
  RUNPOD.md              Step-by-step cloud workflow
src/efficientzero/
  networks/              H, G, F (representation, dynamics, prediction)
  mcts/                  Sampled Gumbel search + tree node
  replay/                Prioritized FIFO buffer + episode storage
  actor/                 Self-play data collection + mixed value target
  training/              Trainer loop, losses, target net
  envs/                  Atari, DM-Control, smoke (CartPole) wrappers
  utils/                 Config loader, W&B logger, device detection
tests/                   Unit + tiny end-to-end tests
```

## Algorithmic notes

The code references specific sections of the EZ-V2 paper in comments,
so it's readable side-by-side with [arXiv:2403.00564](https://arxiv.org/abs/2403.00564).
Key innovations vs MuZero:

| Component | What changed | Where in this repo |
|---|---|---|
| Action selection | Sampled Gumbel search (works for discrete + continuous) | `src/efficientzero/mcts/sampled_gumbel_search.py` |
| Value target | Mixed search-value + TD-bootstrap (handles off-policy data) | `src/efficientzero/actor/mixed_value_target.py` |
| Representation training | Self-supervised consistency loss (SimSiam-style) | `src/efficientzero/training/losses.py` |
| Reward / value heads | Categorical (twohot) instead of scalar | `src/efficientzero/networks/prediction.py` |

## Compute expectations

Per the paper (Appendix J.3), EZ-V2 on a Walker-Run task takes
~2.7 hours per 100k training steps on 8× RTX 3090. On a single 3090,
expect roughly 8× longer per task. A full Atari-100k benchmark
(26 games × 3 seeds) is ~$2k of cloud-GPU time at RunPod prices.
A single task to demonstrate the algorithm working: ~$1–3.

## License

Personal research project. See LICENSE if added.
