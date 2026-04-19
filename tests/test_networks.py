"""Forward-pass shape tests for H, G, F (vector and image, discrete and continuous)."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from efficientzero.envs.spec import EnvSpec
from efficientzero.networks import build_networks


def _net_cfg(hidden: int = 32, blocks: int = 1, vsize: int = 11) -> "OmegaConf":
    return OmegaConf.create(
        {
            "network": {
                "hidden_dim": hidden,
                "num_blocks": blocks,
                "value_support_size": vsize,
                "value_support_min": -10.0,
                "value_support_max": 10.0,
                "reward_support_size": vsize,
                "reward_support_min": -2.0,
                "reward_support_max": 2.0,
                "num_continuous_action_samples": 8,
            }
        }
    )


def test_vector_discrete_shapes():
    spec = EnvSpec(obs_kind="vector", obs_shape=(4,), action_kind="discrete", num_actions=2)
    nets = build_networks(_net_cfg(), spec)
    obs = torch.randn(3, 4)
    out = nets.initial_inference(obs)
    assert out.latent.shape == (3, 32)
    assert out.policy_logits.shape == (3, 2)
    assert out.value.shape == (3, 11)
    actions = torch.tensor([0, 1, 0])
    out2 = nets.recurrent_inference(out.latent, actions)
    assert out2.latent.shape == (3, 32)
    assert out2.reward.shape == (3, 11)


def test_image_continuous_shapes():
    spec = EnvSpec(
        obs_kind="image",
        obs_shape=(3, 32, 32),
        action_kind="continuous",
        action_shape=(2,),
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )
    nets = build_networks(_net_cfg(hidden=32, blocks=1), spec)
    obs = torch.randn(2, 3, 32, 32)
    out = nets.initial_inference(obs)
    assert out.latent.shape == (2, 32)
    assert out.policy_logits.shape == (2, 4)
    actions = torch.tensor([[0.5, -0.5], [0.0, 0.1]])
    out2 = nets.recurrent_inference(out.latent, actions)
    assert out2.latent.shape == (2, 32)
