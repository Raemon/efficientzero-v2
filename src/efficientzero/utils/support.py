"""Categorical (twohot) representation of scalar values, à la C51 / MuZero.

Both value and reward heads predict a distribution over a fixed set of
support atoms; a scalar is recovered as the expectation. This stabilizes
training over wide reward/value ranges. See MuZero appendix F.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Support:
    size: int
    vmin: float
    vmax: float

    @property
    def atoms(self) -> torch.Tensor:
        return torch.linspace(self.vmin, self.vmax, self.size)

    def scalar_to_twohot(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalars as a two-hot distribution over the support atoms."""
        x = x.clamp(self.vmin, self.vmax)
        atoms = self.atoms.to(x.device)
        delta = (self.vmax - self.vmin) / (self.size - 1)
        idx_lo = ((x - self.vmin) / delta).floor().long().clamp(0, self.size - 1)
        idx_hi = (idx_lo + 1).clamp(0, self.size - 1)
        weight_hi = (x - atoms[idx_lo]) / delta
        weight_hi = weight_hi.clamp(0.0, 1.0)
        out = torch.zeros(*x.shape, self.size, device=x.device, dtype=torch.float32)
        out.scatter_(-1, idx_lo.unsqueeze(-1), (1.0 - weight_hi).unsqueeze(-1))
        out.scatter_add_(-1, idx_hi.unsqueeze(-1), weight_hi.unsqueeze(-1))
        return out

    def logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value under the categorical distribution defined by `logits`."""
        probs = F.softmax(logits, dim=-1)
        atoms = self.atoms.to(logits.device)
        return (probs * atoms).sum(dim=-1)
