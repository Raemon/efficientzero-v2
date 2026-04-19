"""Device resolution. `auto` picks cuda > mps > cpu."""
from __future__ import annotations

import torch


def resolve_device(name: str | torch.device) -> torch.device:
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
