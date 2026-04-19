"""Target network: a frozen copy of the online network, periodically refreshed."""
from __future__ import annotations

import copy

import torch.nn as nn


class TargetNetwork:
    def __init__(self, online: nn.Module):
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    def sync(self, online: nn.Module) -> None:
        self.target.load_state_dict(online.state_dict())
        self.target.eval()

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)
