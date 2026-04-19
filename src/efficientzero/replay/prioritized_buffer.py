"""Prioritized FIFO replay buffer over Episode objects.

Sampling unit is a *training sequence* of length L = unroll_steps + 1
starting at some position t inside an episode. We store the
self-supervised consistency target by also fetching obs at position
t + 1 ... t + L (when available).

Priority: per-step priority = |search_value - bootstrap| (set externally),
fall back to max-priority for new transitions. We use simple proportional
sampling. For pedagogical clarity this is O(N) per sample; switch to a
sum-tree if you need throughput.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np

from efficientzero.replay.episode import Episode


@dataclass
class TrainBatchItem:
    obs_seq: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    target_policies: np.ndarray
    target_values: np.ndarray
    target_consistency_obs: np.ndarray
    weight: float
    episode_index: int
    step_index: int
    insertion_index: int


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
        priority_alpha: float = 1.0,
        priority_beta: float = 1.0,
    ):
        self.capacity = int(capacity)
        self.unroll_steps = int(unroll_steps)
        self.td_steps = int(td_steps)
        self.discount = float(discount)
        self.alpha = float(priority_alpha)
        self.beta = float(priority_beta)
        self.episodes: Deque[Episode] = deque()
        self.priorities: Deque[np.ndarray] = deque()
        self._size = 0
        self._max_priority = 1.0
        self._next_insertion_index = 0

    @property
    def size(self) -> int:
        return self._size

    def add_episode(self, ep: Episode) -> None:
        if len(ep) == 0:
            return
        ep.insertion_index = self._next_insertion_index
        self._next_insertion_index += 1
        self.episodes.append(ep)
        self.priorities.append(np.full(len(ep), self._max_priority, dtype=np.float32))
        self._size += len(ep)
        while self._size > self.capacity and self.episodes:
            old = self.episodes.popleft()
            self.priorities.popleft()
            self._size -= len(old)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def sample_batch(self, batch_size: int) -> tuple[List[TrainBatchItem], List[tuple[int, int]]]:
        ep_lens = np.array([len(e) for e in self.episodes], dtype=np.int64)
        ep_probs = ep_lens / ep_lens.sum()
        chosen_eps = np.random.choice(len(self.episodes), size=batch_size, p=ep_probs)

        items: List[TrainBatchItem] = []
        index_pairs: List[tuple[int, int]] = []
        for ep_idx in chosen_eps:
            ep = self.episodes[ep_idx]
            prios = self.priorities[ep_idx] ** self.alpha
            prios = prios / prios.sum()
            t = int(np.random.choice(len(ep), p=prios))
            weight = (1.0 / (len(ep) * prios[t] + 1e-8)) ** self.beta
            items.append(self._make_item(ep, t, weight=weight))
            index_pairs.append((ep_idx, t))

        max_w = max(it.weight for it in items)
        for it in items:
            it.weight = it.weight / max_w
        return items, index_pairs

    def update_priorities(self, index_pairs: List[tuple[int, int]], new_priorities: np.ndarray) -> None:
        for (ep_idx, t), p in zip(index_pairs, new_priorities):
            if ep_idx >= len(self.episodes):
                continue
            self.priorities[ep_idx][t] = float(p) + 1e-6
            self._max_priority = max(self._max_priority, float(p))

    def _make_item(self, ep: Episode, t: int, weight: float) -> TrainBatchItem:
        L = self.unroll_steps
        obs_dim_shape = ep.obs[0].shape
        action_dim_shape = ep.actions[0].shape
        policy_shape = ep.search_policies[0].shape

        obs_seq = np.zeros((L + 1, *obs_dim_shape), dtype=np.float32)
        actions = np.zeros((L, *action_dim_shape), dtype=np.float32)
        rewards = np.zeros((L,), dtype=np.float32)
        target_policies = np.zeros((L + 1, *policy_shape), dtype=np.float32)
        target_values = np.zeros((L + 1,), dtype=np.float32)
        target_cons_obs = np.zeros((L, *obs_dim_shape), dtype=np.float32)

        ep_len = len(ep)
        obs_seq[0] = ep.obs[t]
        target_values[0] = self._n_step_value(ep, t)
        target_policies[0] = ep.search_policies[t]
        for i in range(L):
            j = t + i
            if j < ep_len:
                actions[i] = ep.actions[j]
                rewards[i] = ep.rewards[j]
            if j + 1 < ep_len:
                obs_seq[i + 1] = ep.obs[j + 1]
                target_cons_obs[i] = ep.obs[j + 1]
                target_policies[i + 1] = ep.search_policies[j + 1]
                target_values[i + 1] = self._n_step_value(ep, j + 1)
            else:
                obs_seq[i + 1] = ep.obs[-1]
                target_cons_obs[i] = ep.obs[-1]

        return TrainBatchItem(
            obs_seq=obs_seq,
            actions=actions,
            rewards=rewards,
            target_policies=target_policies,
            target_values=target_values,
            target_consistency_obs=target_cons_obs,
            weight=weight,
            episode_index=ep.insertion_index,
            step_index=t,
            insertion_index=ep.insertion_index,
        )

    def _n_step_value(self, ep: Episode, t: int) -> float:
        """Standard n-step TD target with bootstrap from the search value."""
        n = self.td_steps
        ep_len = len(ep)
        bootstrap_t = t + n
        value = 0.0
        steps_remaining_in_window = list(range(min(n, ep_len - t)))
        for i in steps_remaining_in_window:
            value += (self.discount ** i) * ep.rewards[t + i]
        if bootstrap_t < ep_len:
            value += (self.discount ** n) * ep.search_values[bootstrap_t]
        return float(value)
