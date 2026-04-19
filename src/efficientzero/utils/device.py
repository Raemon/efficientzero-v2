"""Device resolution. `auto` picks cuda > mps > cpu."""
from __future__ import annotations

import torch


def resolve_device(name: str | torch.device) -> torch.device:
    """Default to cuda, fall back to cpu. We deliberately do NOT auto-pick
    MPS (Apple Silicon GPU): some ops in the search/twohot path use float64
    paths that MPS doesn't support, so MPS is a footgun for local dev.
    Pass `device=mps` explicitly if you really want it.
    """
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)
