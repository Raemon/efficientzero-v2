"""Smoke tests for sampled Gumbel search."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from efficientzero.envs.spec import EnvSpec
from efficientzero.mcts import run_search
from efficientzero.networks import build_networks


def _cfg():
    return OmegaConf.create(
        {
            "network": {
                "hidden_dim": 32,
                "num_blocks": 1,
                "value_support_size": 11,
                "value_support_min": -10.0,
                "value_support_max": 10.0,
                "reward_support_size": 11,
                "reward_support_min": -2.0,
                "reward_support_max": 2.0,
                "num_continuous_action_samples": 8,
            },
            "mcts": {
                "num_simulations": 8,
                "num_sampled_actions": 0,
                "pb_c_init": 1.25,
                "pb_c_base": 19652,
                "discount": 0.99,
                "gumbel_scale": 1.0,
                "dirichlet_alpha": 0.3,
                "dirichlet_frac": 0.0,
            },
        }
    )


def test_search_discrete_returns_valid_action():
    spec = EnvSpec(obs_kind="vector", obs_shape=(4,), action_kind="discrete", num_actions=2)
    cfg = _cfg()
    nets = build_networks(cfg, spec)
    obs = torch.randn(1, 4)
    result = run_search(nets, obs, spec, cfg.mcts)
    assert result.recommended_action in (0, 1)
    assert result.policy_target.shape == (2,)
    assert abs(result.policy_target.sum() - 1.0) < 1e-4


def test_search_continuous_returns_in_bounds():
    spec = EnvSpec(
        obs_kind="vector",
        obs_shape=(5,),
        action_kind="continuous",
        action_shape=(3,),
        action_low=(-1.0, -1.0, -1.0),
        action_high=(1.0, 1.0, 1.0),
    )
    cfg = _cfg()
    cfg.mcts.num_sampled_actions = 8
    nets = build_networks(cfg, spec)
    obs = torch.randn(1, 5)
    result = run_search(nets, obs, spec, cfg.mcts)
    assert result.recommended_action.shape == (3,)
    assert (result.recommended_action >= -1.0001).all() and (result.recommended_action <= 1.0001).all()
