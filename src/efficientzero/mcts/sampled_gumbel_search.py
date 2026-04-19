"""Sampled Gumbel Search (EZ-V2 §3.2).

Combines two ideas:

1. **Sampled MCTS** (Hubert et al. 2021): instead of enumerating actions
   (impossible for continuous control), sample K actions from the prior
   policy at each node and treat them as the action set.

2. **Gumbel MuZero** (Danihelka et al. 2021): replace stochastic PUCT
   selection at the root with deterministic Sequential Halving with
   Gumbel, which provides a *policy-improvement guarantee* even with very
   few simulations. Non-root selection uses a deterministic action with
   completed Q-values as the criterion.

What `run_search` returns is a `SearchResult` with:
  - `recommended_action`: action chosen by Gumbel argmax at the root
  - `policy_target`: improved policy (normalized completed-Q logits) used
    as the policy training target
  - `value_target`: search value (mixed reward + bootstrap) used as the
    value training target

This implementation handles BOTH discrete and continuous control via the
same code path; the only difference is how root actions are sampled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from efficientzero.envs.spec import EnvSpec
from efficientzero.mcts.node import Node
from efficientzero.networks import Networks


@dataclass
class SearchResult:
    recommended_action: np.ndarray
    policy_target: np.ndarray
    value_target: float
    sampled_actions: np.ndarray
    root_value: float
    visit_counts: np.ndarray


@torch.no_grad()
def run_search(
    nets: Networks,
    obs: torch.Tensor,
    spec: EnvSpec,
    cfg: DictConfig,
    add_exploration_noise: bool = False,
) -> SearchResult:
    """Run a single sampled-Gumbel search from `obs`.

    `obs` should be a single observation (1, ...) on the right device.
    """
    num_simulations = int(cfg.num_simulations)
    discount = float(cfg.discount)

    initial = nets.initial_inference(obs)
    root_value_scalar = float(nets.value_support.logits_to_scalar(initial.value).item())
    prior_logits = initial.policy_logits.squeeze(0)

    sampled_actions, action_priors_logits = _sample_actions_at_root(
        prior_logits=prior_logits,
        spec=spec,
        cfg=cfg,
        add_exploration_noise=add_exploration_noise,
    )
    K = sampled_actions.shape[0]

    root = Node(latent=initial.latent.squeeze(0))
    for k in range(K):
        root.children[k] = Node(prior=float(F.softmax(action_priors_logits, dim=0)[k].item()))
        root.children[k].sampled_action = sampled_actions[k]

    gumbel = _sample_gumbel(K, device=obs.device)
    sim_budget_per_action = max(1, num_simulations // K)
    for k in range(K):
        for _ in range(sim_budget_per_action):
            _simulate(nets=nets, root=root, action_index=k, spec=spec, discount=discount)

    completed_q = _completed_q_values(root, root_value_scalar)
    sigma_q = _sigma(completed_q)
    score = gumbel.cpu().numpy() + action_priors_logits.cpu().numpy() + sigma_q
    best_k = int(np.argmax(score))
    recommended_action = sampled_actions[best_k]

    policy_target = _improved_policy(action_priors_logits.cpu().numpy(), sigma_q)
    visit_counts = np.array([root.children[k].visit_count for k in range(K)], dtype=np.int64)
    search_value = _root_search_value(root, root_value_scalar)

    return SearchResult(
        recommended_action=recommended_action,
        policy_target=policy_target,
        value_target=float(search_value),
        sampled_actions=sampled_actions,
        root_value=root_value_scalar,
        visit_counts=visit_counts,
    )


def _sample_actions_at_root(
    prior_logits: torch.Tensor,
    spec: EnvSpec,
    cfg: DictConfig,
    add_exploration_noise: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    """Return (K sampled actions array, K logits over those samples).

    For discrete: either all actions (if K==0 or K>=num_actions) or top-K by
    Gumbel-Top-K. For continuous: K iid samples from the Gaussian policy,
    with logits being the per-sample log-prob (used as the prior).
    """
    if spec.action_kind == "discrete":
        return _sample_discrete_root(prior_logits, spec, cfg, add_exploration_noise)
    return _sample_continuous_root(prior_logits, spec, cfg)


def _sample_discrete_root(
    prior_logits: torch.Tensor, spec: EnvSpec, cfg: DictConfig, add_exploration_noise: bool
) -> tuple[np.ndarray, torch.Tensor]:
    K = int(cfg.num_sampled_actions)
    n = spec.num_actions
    logits = prior_logits.clone()
    if add_exploration_noise and float(cfg.dirichlet_frac) > 0.0:
        noise = np.random.dirichlet([float(cfg.dirichlet_alpha)] * n)
        frac = float(cfg.dirichlet_frac)
        probs = (1.0 - frac) * F.softmax(logits, dim=0).cpu().numpy() + frac * noise
        logits = torch.log(torch.tensor(probs, device=logits.device).clamp_min(1e-8))
    if K <= 0 or K >= n:
        actions = np.arange(n, dtype=np.int64)
        return actions, logits
    g = _sample_gumbel(n, device=logits.device)
    topk = torch.topk(g + logits, k=K).indices.cpu().numpy()
    return topk.astype(np.int64), logits[topk]


def _sample_continuous_root(
    prior_logits: torch.Tensor, spec: EnvSpec, cfg: DictConfig
) -> tuple[np.ndarray, torch.Tensor]:
    K = int(cfg.num_sampled_actions)
    if K <= 0:
        K = 16
    d = spec.action_shape[0]
    mean, log_std = prior_logits[:d], prior_logits[d:]
    std = log_std.clamp(-5, 2).exp()
    dist = torch.distributions.Normal(mean, std)
    samples = dist.sample((K,))
    log_probs = dist.log_prob(samples).sum(dim=-1)
    low = torch.tensor(spec.action_low, device=samples.device)
    high = torch.tensor(spec.action_high, device=samples.device)
    samples = torch.clamp(samples, low, high)
    return samples.cpu().numpy(), log_probs


def _simulate(nets: Networks, root: Node, action_index: int, spec: EnvSpec, discount: float) -> None:
    """One simulation rollout from the root through child `action_index`.

    Down the tree we use deterministic completed-Q action selection (Gumbel
    paper §6). At an unexpanded leaf we expand by sampling K children.
    """
    path = [root]
    chosen_indices = [action_index]
    node = root.children[action_index]
    path.append(node)

    while node.expanded:
        next_idx = _select_child(node, root_value_scalar=root.value)
        chosen_indices.append(next_idx)
        node = node.children[next_idx]
        path.append(node)

    parent = path[-2]
    chosen_action = node.sampled_action
    if chosen_action is None:
        chosen_action = parent.children[chosen_indices[-1]].sampled_action
    action_tensor = _action_to_tensor(chosen_action, spec, parent.latent.device)
    out = nets.recurrent_inference(parent.latent.unsqueeze(0), action_tensor.unsqueeze(0))
    node.latent = out.latent.squeeze(0)
    node.reward = float(nets.reward_support.logits_to_scalar(out.reward).item())
    leaf_value = float(nets.value_support.logits_to_scalar(out.value).item())

    prior_logits = out.policy_logits.squeeze(0)
    samples, sample_logits = _sample_actions_at_inner(prior_logits, spec, K=len(parent.children))
    for k, a in enumerate(samples):
        child = Node(prior=float(F.softmax(sample_logits, dim=0)[k].item()))
        child.sampled_action = a
        node.children[k] = child

    _backpropagate(path, leaf_value, discount)


def _sample_actions_at_inner(prior_logits: torch.Tensor, spec: EnvSpec, K: int) -> tuple[np.ndarray, torch.Tensor]:
    if spec.action_kind == "discrete":
        n = spec.num_actions
        if K >= n:
            return np.arange(n, dtype=np.int64), prior_logits
        g = _sample_gumbel(n, device=prior_logits.device)
        topk = torch.topk(g + prior_logits, k=K).indices.cpu().numpy()
        return topk.astype(np.int64), prior_logits[topk]
    d = spec.action_shape[0]
    mean, log_std = prior_logits[:d], prior_logits[d:]
    std = log_std.clamp(-5, 2).exp()
    dist = torch.distributions.Normal(mean, std)
    samples = dist.sample((K,))
    log_probs = dist.log_prob(samples).sum(dim=-1)
    low = torch.tensor(spec.action_low, device=samples.device)
    high = torch.tensor(spec.action_high, device=samples.device)
    samples = torch.clamp(samples, low, high)
    return samples.cpu().numpy(), log_probs


def _select_child(node: Node, root_value_scalar: float) -> int:
    """Deterministic action selection at non-root nodes (Gumbel paper §6).

    Pick action a* = argmax_a (pi'(a) - N(a) / (1 + sum_b N(b))) where pi'
    is the improved policy from completed Q-values.
    """
    completed_q = _completed_q_values(node, root_value_scalar)
    sigma_q = _sigma(completed_q)
    K = len(node.children)
    priors = np.array([node.children[k].prior for k in range(K)], dtype=np.float32)
    log_priors = np.log(priors.clip(min=1e-8))
    pi_prime = _softmax(log_priors + sigma_q)
    visits = np.array([node.children[k].visit_count for k in range(K)], dtype=np.float32)
    score = pi_prime - visits / (1.0 + visits.sum())
    return int(np.argmax(score))


def _backpropagate(path: list[Node], leaf_value: float, discount: float) -> None:
    g = leaf_value
    for node in reversed(path):
        node.value_sum += g
        node.visit_count += 1
        g = node.reward + discount * g


def _completed_q_values(node: Node, root_value_scalar: float) -> np.ndarray:
    """For unvisited children, complete their Q with the parent's value (Danihelka §4)."""
    K = len(node.children)
    out = np.zeros(K, dtype=np.float32)
    for k in range(K):
        child = node.children[k]
        if child.visit_count > 0:
            out[k] = child.reward + child.value
        else:
            out[k] = root_value_scalar
    return out


def _sigma(q: np.ndarray) -> np.ndarray:
    """Monotonic transform sigma(q) used by Gumbel MuZero. Linear is fine."""
    return q.astype(np.float32)


def _improved_policy(prior_logits: np.ndarray, sigma_q: np.ndarray) -> np.ndarray:
    return _softmax(prior_logits + sigma_q)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _sample_gumbel(n: int, device: torch.device) -> torch.Tensor:
    u = torch.rand(n, device=device).clamp(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(u))


def _action_to_tensor(action: np.ndarray, spec: EnvSpec, device: torch.device) -> torch.Tensor:
    if spec.action_kind == "discrete":
        return torch.tensor(int(action), device=device)
    return torch.tensor(np.asarray(action, dtype=np.float32), device=device)


def _root_search_value(root: Node, fallback: float) -> float:
    if root.visit_count == 0:
        return fallback
    return root.value
