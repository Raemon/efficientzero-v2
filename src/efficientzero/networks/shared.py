"""Building blocks shared across H, G, F."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers_in_order: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers_in_order.append(nn.Linear(prev, hidden))
            layers_in_order.append(nn.LayerNorm(hidden))
            layers_in_order.append(nn.SiLU())
            prev = hidden
        layers_in_order.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers_in_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return F.silu(x + h)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(x + h)


class ProjectionHead(nn.Module):
    """SimSiam-style projection / predictor MLP, used by the consistency loss."""

    def __init__(self, in_dim: int, hidden: int, predictor: bool = False):
        super().__init__()
        if predictor:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
