"""Headless World 1-1 environment for Stable-Baselines3."""

import cv2
import gymnasium as gym
import numpy as np
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack


class MarioPPOEnv(gym.Env):
    frame_skip = 4
    frame_size = 84
    flag_bonus = 50.0

    def __init__(self):
        cv2.setNumThreads(1)
        # Use the NES environment directly so Gym's legacy TimeLimit does not
        # sit between the emulator's four-value step and the Gymnasium adapter.
        self.emulator = SuperMarioBrosEnv(target=(1, 1))
        self.controller = JoypadSpace(self.emulator, SIMPLE_MOVEMENT)
        self.action_space = gym.spaces.Discrete(self.controller.action_space.n)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(1, self.frame_size, self.frame_size), dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        frame = self.emulator.reset(seed=seed, options=options)
        return self.preprocess(frame), {}

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            frame, reward, terminated, info = self.controller.step(int(action))
            total_reward += reward
            if terminated:
                break

        if info["flag_get"]:
            total_reward += self.flag_bonus
        return self.preprocess(frame), total_reward, terminated, False, info

    def preprocess(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            grayscale, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA
        )
        return resized[np.newaxis, ...]

    def close(self):
        self.controller.close()


def make_episode_env():
    return Monitor(gym.wrappers.TimeLimit(MarioPPOEnv(), max_episode_steps=1500))


def make_ppo_env(n_envs=1):
    """Give training and evaluation identical inputs and episode limits."""
    factories = [make_episode_env] * n_envs
    if n_envs == 1:
        env = DummyVecEnv(factories)
    else:
        env = SubprocVecEnv(factories, start_method="spawn")
    return VecFrameStack(env, n_stack=4, channels_order="first")
