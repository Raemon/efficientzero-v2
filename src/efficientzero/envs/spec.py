"""Environment specification: shape and kind metadata used by networks/MCTS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EnvSpec:
    obs_kind: Literal["vector", "image"]
    obs_shape: tuple[int, ...]
    action_kind: Literal["discrete", "continuous"]
    num_actions: int = 0
    action_shape: tuple[int, ...] = ()
    action_low: tuple[float, ...] = ()
    action_high: tuple[float, ...] = ()
