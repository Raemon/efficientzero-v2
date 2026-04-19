"""Trainer: collects self-play episodes and runs SGD on a replay buffer.

Outer loop:
    while not done:
        collect K env steps via self-play          (actor.self_play.collect_episode)
        for each gradient step until next collect:
            sample batch from replay
            unroll networks for L steps
            compute losses (reward / value / policy / consistency / entropy)
            backward + optim.step()
        periodically: sync target net, eval, log, checkpoint

For pedagogical clarity self-play and learning run *synchronously in the
same process* (no async actors). For sample-efficiency benchmarks like
Atari 100k that's perfectly fine — the real wall-clock cost is the
gradient steps, not the env steps.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from efficientzero.actor.mixed_value_target import staleness_weight
from efficientzero.actor.self_play import collect_episode, play_episodes
from efficientzero.envs import make_env
from efficientzero.networks import build_networks
from efficientzero.replay.prioritized_buffer import PrioritizedReplay, TrainBatchItem
from efficientzero.training.losses import (
    categorical_loss,
    consistency_loss,
    policy_entropy,
    policy_loss,
)
from efficientzero.training.target_network import TargetNetwork
from efficientzero.utils.config import save_config
from efficientzero.utils.device import resolve_device
from efficientzero.utils.logging import Logger


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        self.device = resolve_device(str(cfg.device))

        self.env = make_env(cfg.env, seed=int(cfg.seed))
        self.eval_env = make_env(cfg.env, seed=int(cfg.seed) + 10_000)
        self.spec = self.env.spec

        self.nets = build_networks(cfg, self.spec).to(self.device)
        self.target = TargetNetwork(self.nets)

        self.optimizer = self._make_optimizer()
        self.replay = PrioritizedReplay(
            capacity=int(cfg.replay.capacity),
            unroll_steps=int(cfg.replay.unroll_steps),
            td_steps=int(cfg.replay.td_steps),
            discount=float(cfg.mcts.discount),
            priority_alpha=float(cfg.replay.priority_alpha),
            priority_beta=float(cfg.replay.priority_beta),
        )
        self.logger = Logger(cfg, run_name=str(cfg.name))
        self.train_step = 0
        self.env_steps = 0
        self.checkpoint_dir = Path(cfg.trainer.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.checkpoint_dir / "config.yaml")

    def _make_optimizer(self) -> torch.optim.Optimizer:
        opt_name = str(self.cfg.trainer.optimizer).lower()
        params = self.nets.parameters()
        lr = float(self.cfg.trainer.learning_rate)
        wd = float(self.cfg.trainer.weight_decay)
        if opt_name == "adam":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if opt_name == "sgd":
            return torch.optim.SGD(
                params, lr=lr, weight_decay=wd, momentum=float(self.cfg.trainer.get("sgd_momentum", 0.9))
            )
        raise ValueError(f"Unknown optimizer: {opt_name}")

    def run(self) -> None:
        total_steps = int(self.cfg.trainer.total_train_steps)
        warmup = int(self.cfg.actor.warmup_transitions)
        collect_per_iter = int(self.cfg.actor.collect_per_iter)
        log_interval = int(self.cfg.trainer.log_interval)
        eval_interval = int(self.cfg.trainer.eval_interval)
        target_update_interval = int(self.cfg.trainer.target_update_interval)
        checkpoint_interval = int(self.cfg.trainer.checkpoint_interval)
        batch_size = int(self.cfg.trainer.batch_size)

        progress = tqdm(total=total_steps, desc=str(self.cfg.name))
        last_log_time = time.time()
        while self.replay.size < warmup:
            self._collect_one()

        while self.train_step < total_steps:
            self._collect_one()
            steps_this_iter = max(1, collect_per_iter)
            for _ in range(steps_this_iter):
                if not self.replay.can_sample(batch_size):
                    break
                metrics = self._train_step(batch_size)
                self.train_step += 1
                progress.update(1)

                if self.train_step % target_update_interval == 0:
                    self.target.sync(self.nets)
                if self.train_step % log_interval == 0:
                    sps = log_interval / max(1e-6, time.time() - last_log_time)
                    last_log_time = time.time()
                    self.logger.log({**metrics, "train/steps_per_sec": sps, "env/steps": self.env_steps}, step=self.train_step)
                if self.train_step % eval_interval == 0:
                    eval_returns = play_episodes(
                        nets=self.nets,
                        env=self.eval_env,
                        mcts_cfg=self.cfg.mcts,
                        num_episodes=int(self.cfg.trainer.eval_episodes),
                        device=self.device,
                        deterministic=True,
                    )
                    self.logger.log(
                        {"eval/return_mean": float(np.mean(eval_returns)), "eval/return_max": float(np.max(eval_returns))},
                        step=self.train_step,
                    )
                if self.train_step % checkpoint_interval == 0:
                    self._save_checkpoint()
                if self.train_step >= total_steps:
                    break

        progress.close()
        self._save_checkpoint(name="latest.pt")
        self.logger.finish()

    def _collect_one(self) -> None:
        ep = collect_episode(
            nets=self.nets,
            env=self.env,
            mcts_cfg=self.cfg.mcts,
            device=self.device,
            add_exploration_noise=True,
        )
        self.env_steps += len(ep)
        self.replay.add_episode(ep)

    def _train_step(self, batch_size: int) -> dict[str, float]:
        items, index_pairs = self.replay.sample_batch(batch_size)
        batch = self._stack_items(items)

        coefs = self.cfg.trainer.loss_coefs
        L = int(self.cfg.replay.unroll_steps)
        T1 = int(self.cfg.trainer.mixed_value_target_T1)
        T2 = int(self.cfg.trainer.mixed_value_target_T2)
        max_insertion = self.replay._next_insertion_index - 1

        weights_per_step = torch.tensor(
            [staleness_weight(self.train_step, it.insertion_index, T1, T2, max_insertion) for it in items],
            dtype=torch.float32,
            device=self.device,
        )

        out = self.nets.initial_inference(batch["obs0"])
        td_value_targets = batch["target_values"]
        sve_value_targets = td_value_targets
        mixed_value = weights_per_step.unsqueeze(1) * td_value_targets + (1.0 - weights_per_step.unsqueeze(1)) * sve_value_targets

        l_value = categorical_loss(out.value, self.nets.value_support.scalar_to_twohot(mixed_value[:, 0]))
        l_policy = policy_loss(out.policy_logits, batch["target_policies"][:, 0])
        l_reward = torch.zeros_like(l_value)
        l_consistency = torch.zeros_like(l_value)
        entropy = policy_entropy(out.policy_logits)

        latent = out.latent
        for i in range(L):
            action = batch["actions"][:, i]
            step_out = self.nets.recurrent_inference(latent, action)
            latent = step_out.latent
            l_reward = l_reward + categorical_loss(
                step_out.reward, self.nets.reward_support.scalar_to_twohot(batch["rewards"][:, i])
            )
            l_value = l_value + categorical_loss(
                step_out.value, self.nets.value_support.scalar_to_twohot(mixed_value[:, i + 1])
            )
            l_policy = l_policy + policy_loss(step_out.policy_logits, batch["target_policies"][:, i + 1])
            entropy = entropy + policy_entropy(step_out.policy_logits)

            with torch.no_grad():
                target_repr = self.target.target.representation(batch["target_obs"][:, i])
                target_proj = self.target.target.project(target_repr, with_predictor=False)
            pred_proj = self.nets.project(latent, with_predictor=True)
            l_consistency = l_consistency + consistency_loss(pred_proj, target_proj)

        weights_is = batch["weights"]
        total = (
            float(coefs.reward) * l_reward
            + float(coefs.value) * l_value
            + float(coefs.policy) * l_policy
            + float(coefs.consistency) * l_consistency
            - float(self.cfg.trainer.policy_entropy_coef) * entropy
        )
        loss = (total * weights_is).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nets.parameters(), float(self.cfg.trainer.grad_clip))
        self.optimizer.step()

        with torch.no_grad():
            new_priorities = (l_value.detach()).abs().cpu().numpy() + 1e-3
        self.replay.update_priorities(index_pairs, new_priorities)

        return {
            "loss/total": float(loss.item()),
            "loss/reward": float(l_reward.mean().item()),
            "loss/value": float(l_value.mean().item()),
            "loss/policy": float(l_policy.mean().item()),
            "loss/consistency": float(l_consistency.mean().item()),
            "loss/entropy": float(entropy.mean().item()),
            "replay/size": int(self.replay.size),
        }

    def _stack_items(self, items: list[TrainBatchItem]) -> dict[str, torch.Tensor]:
        obs_seq = np.stack([it.obs_seq for it in items], axis=0)
        actions = np.stack([it.actions for it in items], axis=0)
        rewards = np.stack([it.rewards for it in items], axis=0)
        target_policies = np.stack([it.target_policies for it in items], axis=0)
        target_values = np.stack([it.target_values for it in items], axis=0)
        target_obs = np.stack([it.target_consistency_obs for it in items], axis=0)
        weights = np.array([it.weight for it in items], dtype=np.float32)

        action_tensor = torch.from_numpy(actions).to(self.device)
        if self.spec.action_kind == "discrete":
            action_tensor = action_tensor.squeeze(-1).long()

        return {
            "obs0": torch.from_numpy(obs_seq[:, 0]).to(self.device),
            "actions": action_tensor,
            "rewards": torch.from_numpy(rewards).to(self.device),
            "target_policies": torch.from_numpy(target_policies).to(self.device),
            "target_values": torch.from_numpy(target_values).to(self.device),
            "target_obs": torch.from_numpy(target_obs).to(self.device),
            "weights": torch.from_numpy(weights).to(self.device),
        }

    def _save_checkpoint(self, name: str | None = None) -> None:
        path = self.checkpoint_dir / (name or f"step_{self.train_step}.pt")
        torch.save(
            {
                "networks": self.nets.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step": self.train_step,
                "env_steps": self.env_steps,
            },
            path,
        )
