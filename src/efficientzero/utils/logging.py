"""Tiny logging shim: prints + optional W&B mirror."""
from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


class Logger:
    def __init__(self, cfg: DictConfig, run_name: str):
        self.run_name = run_name
        self._wandb = None
        if bool(cfg.logging.wandb):
            import wandb

            self._wandb = wandb.init(
                project=cfg.logging.project,
                group=str(cfg.logging.get("group", "default")),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        line = " ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        print(f"[step {step:>7}] {line}", flush=True)
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
