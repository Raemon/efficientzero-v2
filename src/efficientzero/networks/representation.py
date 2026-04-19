"""Representation network H: observation -> latent state s_0.

Outputs a flat (batch, hidden_dim) latent. The original MuZero keeps a
small spatial latent map for image inputs; we pool to a vector instead
to keep G and F as simple MLPs. This costs some expressiveness but is
much easier to read.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from efficientzero.envs.spec import EnvSpec
from efficientzero.networks.shared import (
    ConvBlock,
    MLP,
    ResidualConvBlock,
    ResidualMLPBlock,
)


class RepresentationNet(nn.Module):
    def __init__(self, spec: EnvSpec, net_cfg):
        super().__init__()
        self.spec = spec
        self.hidden_dim = int(net_cfg.hidden_dim)
        self.num_blocks = int(net_cfg.num_blocks)
        self.is_image = spec.obs_kind == "image"

        if not self.is_image:
            self.encoder = MLP(spec.obs_shape[0], self.hidden_dim, self.hidden_dim, depth=2)
            self.trunk = nn.Sequential(*[ResidualMLPBlock(self.hidden_dim) for _ in range(self.num_blocks)])
        else:
            in_ch = spec.obs_shape[0]
            self.encoder = nn.Sequential(
                ConvBlock(in_ch, 64, stride=2),
                ConvBlock(64, 128, stride=2),
                ConvBlock(128, self.hidden_dim, stride=2),
            )
            self.trunk = nn.Sequential(
                *[ResidualConvBlock(self.hidden_dim) for _ in range(self.num_blocks)],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(self.hidden_dim),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        return self.trunk(h)
