"""Episode container: lists of per-step quantities collected during self-play."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Episode:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    search_policies: List[np.ndarray] = field(default_factory=list)
    search_values: List[float] = field(default_factory=list)
    sampled_actions: List[np.ndarray] = field(default_factory=list)
    insertion_index: int = 0
    return_total: float = 0.0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        search_policy: np.ndarray,
        search_value: float,
        sampled_actions: np.ndarray | None = None,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.search_policies.append(search_policy)
        self.search_values.append(search_value)
        if sampled_actions is not None:
            self.sampled_actions.append(sampled_actions)
        self.return_total += reward

    def __len__(self) -> int:
        return len(self.actions)
