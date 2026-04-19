"""End-to-end test: run a tiny smoke training and assert it doesn't crash."""
from __future__ import annotations

from pathlib import Path

import pytest

from efficientzero.training.trainer import Trainer
from efficientzero.utils.config import load_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.mark.timeout(300)
def test_smoke_trains_a_few_steps(tmp_path):
    cfg = load_config(
        CONFIG_PATH,
        overrides=[
            "trainer.total_train_steps=20",
            "actor.warmup_transitions=64",
            "actor.collect_per_iter=8",
            "trainer.batch_size=16",
            "trainer.log_interval=10",
            "trainer.eval_interval=20",
            "trainer.eval_episodes=1",
            "trainer.checkpoint_interval=20",
            "mcts.num_simulations=4",
            f"trainer.checkpoint_dir={tmp_path / 'ckpt'}",
            "logging.wandb=false",
        ],
    )
    trainer = Trainer(cfg)
    trainer.run()
    assert trainer.train_step >= 20
    assert any(p.suffix == ".pt" for p in (tmp_path / "ckpt").glob("*"))
