"""Dynamics network G: (latent s_t, action a_t) -> (latent s_{t+1}, reward distribution)."""
from __future__ import annotations

import torch
import torch.nn as nn

from efficientzero.envs.spec import EnvSpec
from efficientzero.networks.shared import MLP, ResidualMLPBlock


class DynamicsNet(nn.Module):
    def __init__(self, spec: EnvSpec, net_cfg, reward_support_size: int):
        super().__init__()
        self.spec = spec
        self.hidden_dim = int(net_cfg.hidden_dim)
        self.num_blocks = int(net_cfg.num_blocks)

        if spec.action_kind == "discrete":
            self.action_dim = spec.num_actions
            self.action_embed = nn.Embedding(spec.num_actions, self.hidden_dim)
        else:
            self.action_dim = spec.action_shape[0]
            self.action_embed = nn.Linear(self.action_dim, self.hidden_dim)

        self.fuse = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.trunk = nn.Sequential(*[ResidualMLPBlock(self.hidden_dim) for _ in range(self.num_blocks)])
        self.reward_head = MLP(self.hidden_dim, self.hidden_dim, reward_support_size, depth=1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.action_kind == "discrete":
            action = action.long()
            a = self.action_embed(action)
        else:
            a = self.action_embed(action.float())
        h = torch.cat([latent, a], dim=-1)
        h = self.fuse(h)
        next_latent = self.trunk(h)
        reward_logits = self.reward_head(next_latent)
        return next_latent, reward_logits
