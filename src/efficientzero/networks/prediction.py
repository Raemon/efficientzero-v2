"""Prediction network F: latent state -> (policy, value distribution).

For discrete actions: policy is logits over actions (size = num_actions).
For continuous actions: policy is concatenated [mean, log_std] of a
diagonal Gaussian over actions (size = 2 * action_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from efficientzero.envs.spec import EnvSpec
from efficientzero.networks.shared import MLP


class PredictionNet(nn.Module):
    def __init__(self, spec: EnvSpec, net_cfg, value_support_size: int):
        super().__init__()
        self.spec = spec
        self.hidden_dim = int(net_cfg.hidden_dim)

        if spec.action_kind == "discrete":
            policy_out = spec.num_actions
        else:
            policy_out = 2 * spec.action_shape[0]
        self.policy_head = MLP(self.hidden_dim, self.hidden_dim, policy_out, depth=1)
        self.value_head = MLP(self.hidden_dim, self.hidden_dim, value_support_size, depth=1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_head(latent), self.value_head(latent)
