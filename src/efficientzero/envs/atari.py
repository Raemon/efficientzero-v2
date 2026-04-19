"""Atari env factory using gymnasium.wrappers.AtariPreprocessing + FrameStack.

Requires the `[atari]` extra. Standard preprocessing per the Atari 100k
benchmark: action_repeat=4, grayscale 84x84, episodic life, reward clip.
"""
from __future__ import annotations

from omegaconf import DictConfig

from efficientzero.envs.smoke import GymEnvWrapper


def make_atari_env(cfg: DictConfig, seed: int) -> GymEnvWrapper:
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStack

    base = gym.make(str(cfg.id), frameskip=1, repeat_action_probability=0.0, full_action_space=False)
    env = AtariPreprocessing(
        base,
        noop_max=int(cfg.get("noop_max", 30)),
        frame_skip=int(cfg.get("action_repeat", 4)),
        screen_size=int(cfg.resize[0]),
        terminal_on_life_loss=bool(cfg.get("episodic_life", True)),
        grayscale_obs=bool(cfg.get("grayscale", True)),
        scale_obs=True,
    )
    env = FrameStack(env, num_stack=int(cfg.get("frame_stack", 4)))
    if bool(cfg.get("clip_reward", True)):
        env = _ClipRewardWrapper(env)
    return GymEnvWrapper(env, seed=seed)


class _ClipRewardWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        clipped = float(max(-1.0, min(1.0, float(reward))))
        return obs, clipped, terminated, truncated, info

    def close(self):
        self.env.close()
