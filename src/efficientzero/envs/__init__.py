"""Environment factory dispatching on `cfg.env.kind`."""
from __future__ import annotations

from omegaconf import DictConfig

from efficientzero.envs.spec import EnvSpec


def make_env(env_cfg: DictConfig, seed: int):
    kind = str(env_cfg.kind)
    if kind in ("gym", "smoke"):
        from efficientzero.envs.smoke import make_smoke_env

        return make_smoke_env(env_cfg, seed)
    if kind == "atari":
        from efficientzero.envs.atari import make_atari_env

        return make_atari_env(env_cfg, seed)
    if kind == "dmc":
        from efficientzero.envs.dmcontrol import make_dmc_env

        return make_dmc_env(env_cfg, seed)
    raise ValueError(f"Unknown env.kind: {kind}")


__all__ = ["make_env", "EnvSpec"]
