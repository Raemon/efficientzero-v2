"""MCTS tree node.

For continuous control we cannot enumerate all actions; instead the root
stores K *sampled* actions and each child is keyed by the sample index.
For discrete control children are keyed by the action id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch


@dataclass
class Node:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    latent: torch.Tensor | None = None
    children: Dict[int, "Node"] = field(default_factory=dict)
    sampled_action: np.ndarray | None = None

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0
