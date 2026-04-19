"""Load a checkpoint and run evaluation episodes."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from efficientzero.actor.self_play import play_episodes
from efficientzero.envs import make_env
from efficientzero.networks import build_networks
from efficientzero.utils.config import load_config
from efficientzero.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an EfficientZero V2 checkpoint.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    env = make_env(cfg.env, seed=cfg.seed + 1000)
    nets = build_networks(cfg, env.spec).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    nets.load_state_dict(state["networks"])
    nets.eval()

    returns = play_episodes(
        nets=nets,
        env=env,
        mcts_cfg=cfg.mcts,
        num_episodes=args.episodes,
        device=device,
        deterministic=True,
    )
    mean = sum(returns) / len(returns)
    print(f"episodes={len(returns)}  mean_return={mean:.2f}  returns={returns}")


if __name__ == "__main__":
    main()
