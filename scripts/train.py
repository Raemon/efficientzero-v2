"""Entry point for training. Loads a config and hands off to the Trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from efficientzero.training.trainer import Trainer
from efficientzero.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientZero V2.")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    p.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted overrides, e.g. trainer.total_train_steps=1000 seed=42",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
