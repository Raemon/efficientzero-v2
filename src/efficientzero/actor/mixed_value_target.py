"""Mixed value target / SVE (EZ-V2 §3.3).

Standard MuZero uses an n-step TD target with a bootstrap from the *online*
network's value of a future state. With a stale replay buffer the bootstrap
becomes biased. EfficientZero V2 mitigates this by mixing two estimators:

  - Search Value Estimation (SVE): root value from MCTS at the *fresh*
    moment the data was collected (already stored in the episode).
  - n-step TD: standard reward + discount * V(target_net(obs_{t+n})).

The mixing weight depends on data staleness: very fresh data (insertion
index close to the current training step) leans on TD; very stale data
leans on SVE. The crossover is parameterized by T1 (start step) and T2
(staleness window).

This module computes that mixing weight; the buffer already provides the
n-step term and the trainer provides the SVE term as `search_value` from
self-play.
"""
from __future__ import annotations

import numpy as np


def staleness_weight(
    train_step: int,
    insertion_index: int,
    T1: int,
    T2: int,
    max_insertion_index: int,
) -> float:
    """Return alpha in [0, 1] for the TD-bootstrap term; (1 - alpha) goes to SVE.

    Heuristic that matches the paper's qualitative behavior:
      - During warmup (train_step < T1), use TD.
      - After T1, use TD only on data inserted within the last T2 episodes.
    """
    if train_step < T1:
        return 1.0
    staleness_in_episodes = max(0, max_insertion_index - insertion_index)
    if staleness_in_episodes <= T2:
        return 1.0
    return float(np.clip(T2 / max(1, staleness_in_episodes), 0.0, 1.0))


def mix(
    td_target: np.ndarray,
    sve_target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weights = weights.astype(np.float32)
    return weights * td_target + (1.0 - weights) * sve_target
