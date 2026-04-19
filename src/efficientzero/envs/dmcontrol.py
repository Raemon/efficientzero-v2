"""DM-Control env factory using shimmy's DmControlCompatibilityV0.

Requires the `[dmc]` extra. For vision: returns (3*frame_stack, H, W) uint8 frames.
"""
from __future__ import annotations

from omegaconf import DictConfig

from efficientzero.envs.smoke import GymEnvWrapper


def make_dmc_env(cfg: DictConfig, seed: int) -> GymEnvWrapper:
    import gymnasium as gym
    from dm_control import suite
    from shimmy.dm_control_compatibility import DmControlCompatibilityV0

    domain, task = str(cfg.id).split("-", 1)
    dm_env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
    env = DmControlCompatibilityV0(dm_env)

    if str(cfg.obs_kind) == "vector":
        env = _FlattenObsWrapper(env)
    else:
        from gymnasium.wrappers import PixelObservationWrapper, ResizeObservation, GrayScaleObservation, FrameStack

        env = PixelObservationWrapper(env, pixels_only=True)
        env = ResizeObservation(env, shape=tuple(cfg.resize))
        env = FrameStack(env, num_stack=int(cfg.get("frame_stack", 3)))

    if int(cfg.get("action_repeat", 1)) > 1:
        env = _ActionRepeatWrapper(env, repeat=int(cfg.action_repeat))
    return GymEnvWrapper(env, seed=seed)


class _FlattenObsWrapper:
    def __init__(self, env):
        import gymnasium as gym
        import numpy as np

        self.env = env
        self.action_space = env.action_space
        sample, _ = env.reset()
        flat = self._flatten(sample)
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=flat.shape)

    @staticmethod
    def _flatten(obs):
        import numpy as np

        if isinstance(obs, dict):
            return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in obs.values()])
        return np.asarray(obs, dtype=np.float32).ravel()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._flatten(obs), r, term, trunc, info

    def close(self):
        self.env.close()


class _ActionRepeatWrapper:
    def __init__(self, env, repeat: int):
        self.env = env
        self.repeat = repeat
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total = 0.0
        for _ in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(action)
            total += float(r)
            if term or trunc:
                break
        return obs, total, term, trunc, info

    def close(self):
        self.env.close()
