"""Smoke environment: a thin wrapper around any classic Gymnasium env.

We expose a uniform `reset()` / `step(action)` API plus a `.spec` field
holding an EnvSpec, so the rest of the code is env-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

from efficientzero.envs.spec import EnvSpec


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class GymEnvWrapper:
    """Uniform interface over a Gymnasium env."""

    def __init__(self, env: gym.Env, seed: int):
        self.env = env
        self._seed = seed
        self.spec = self._make_spec()

    def _make_spec(self) -> EnvSpec:
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1:
            obs_kind = "vector"
            obs_shape = tuple(obs_space.shape)
        elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            obs_kind = "image"
            obs_shape = tuple(obs_space.shape)
        else:
            raise ValueError(f"Unsupported obs space: {obs_space}")
        if isinstance(act_space, gym.spaces.Discrete):
            return EnvSpec(
                obs_kind=obs_kind,
                obs_shape=obs_shape,
                action_kind="discrete",
                num_actions=int(act_space.n),
            )
        if isinstance(act_space, gym.spaces.Box):
            return EnvSpec(
                obs_kind=obs_kind,
                obs_shape=obs_shape,
                action_kind="continuous",
                action_shape=tuple(act_space.shape),
                action_low=tuple(map(float, act_space.low)),
                action_high=tuple(map(float, act_space.high)),
            )
        raise ValueError(f"Unsupported action space: {act_space}")

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset(seed=self._seed)
        self._seed += 1
        return np.asarray(obs, dtype=np.float32)

    def step(self, action) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(
            obs=np.asarray(obs, dtype=np.float32),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        self.env.close()


def make_smoke_env(cfg: DictConfig, seed: int) -> GymEnvWrapper:
    env = gym.make(str(cfg.id))
    return GymEnvWrapper(env, seed=seed)
