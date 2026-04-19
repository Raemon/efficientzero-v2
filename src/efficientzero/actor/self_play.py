"""Synchronous self-play: rolls out one episode at a time using MCTS."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from efficientzero.envs.smoke import GymEnvWrapper
from efficientzero.mcts import run_search
from efficientzero.networks import Networks
from efficientzero.replay.episode import Episode


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _policy_target_full(search_policy: np.ndarray, num_actions: int, sampled: np.ndarray) -> np.ndarray:
    """For discrete actions, scatter the sampled-action policy into the full action vector."""
    out = np.zeros(num_actions, dtype=np.float32)
    out[sampled.astype(np.int64)] = search_policy
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def collect_episode(
    nets: Networks,
    env: GymEnvWrapper,
    mcts_cfg: DictConfig,
    device: torch.device,
    add_exploration_noise: bool = True,
    max_steps: int = 100_000,
) -> Episode:
    nets.eval()
    spec = env.spec
    obs = env.reset()
    ep = Episode()

    for _ in range(max_steps):
        result = run_search(
            nets=nets,
            obs=_obs_to_tensor(obs, device),
            spec=spec,
            cfg=mcts_cfg,
            add_exploration_noise=add_exploration_noise,
        )

        if spec.action_kind == "discrete":
            action_for_env = int(result.recommended_action)
            visit_probs = result.visit_counts.astype(np.float32)
            visit_probs = visit_probs / max(1.0, visit_probs.sum())
            policy_target = _policy_target_full(visit_probs, spec.num_actions, result.sampled_actions)
            action_stored = np.array([action_for_env], dtype=np.int64)
            sampled_actions_stored = result.sampled_actions.astype(np.int64)
        else:
            action_for_env = np.asarray(result.recommended_action, dtype=np.float32)
            policy_target = result.policy_target.astype(np.float32)
            action_stored = action_for_env.copy()
            sampled_actions_stored = result.sampled_actions.astype(np.float32)

        step = env.step(action_for_env)
        ep.add(
            obs=np.asarray(obs, dtype=np.float32),
            action=action_stored,
            reward=step.reward,
            search_policy=policy_target,
            search_value=result.value_target,
            sampled_actions=sampled_actions_stored,
        )
        obs = step.obs
        if step.terminated or step.truncated:
            break

    return ep


def play_episodes(
    nets: Networks,
    env: GymEnvWrapper,
    mcts_cfg: DictConfig,
    num_episodes: int,
    device: torch.device,
    deterministic: bool = True,
) -> List[float]:
    returns: List[float] = []
    for _ in range(num_episodes):
        ep = collect_episode(
            nets=nets,
            env=env,
            mcts_cfg=mcts_cfg,
            device=device,
            add_exploration_noise=not deterministic,
        )
        returns.append(ep.return_total)
    return returns
