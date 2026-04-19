"""Per-step loss terms.

EZ-V2 minimizes a weighted sum of:
  L = lambda_r * L_reward + lambda_p * L_policy + lambda_v * L_value
      + lambda_c * L_consistency + entropy_coef * (-H[pi])

`reward` and `value` losses are cross-entropies against twohot encodings.
`policy` is cross-entropy against the search-improved policy.
`consistency` is the SimSiam-style negative cosine similarity between the
predicted next latent and the projection of the actual next observation
encoded by the *target* network (stop-gradient through the target side).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def categorical_loss(logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_distribution * log_probs).sum(dim=-1)


def policy_loss(logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1)


def policy_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def consistency_loss(predicted_latent_proj: torch.Tensor, target_latent_proj: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity, à la SimSiam (no l2 normalization here
    because we apply F.normalize internally)."""
    p = F.normalize(predicted_latent_proj, dim=-1)
    z = F.normalize(target_latent_proj.detach(), dim=-1)
    return -(p * z).sum(dim=-1)
