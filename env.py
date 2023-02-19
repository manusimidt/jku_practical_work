import gym
import stable_baselines3.common.env_checker
import numpy as np
from gym import spaces
from enum import Enum

import augmentations

POSSIBLE_AUGMENTATIONS = [
    {'func': augmentations.random_translate, 'params': {'size': 70}},
    {'func': augmentations.random_translate, 'params': {'size': 80}},
    {'func': augmentations.random_translate, 'params': {'size': 90}},
    {'func': augmentations.random_crop, 'params': {'out': 55}},
    {'func': augmentations.random_crop, 'params': {'out': 50}},
    {'func': augmentations.random_crop, 'params': {'out': 45}},
    {'func': augmentations.random_cutout, 'params': {'min_cut': 2, 'max_cut': 5}},
    {'func': augmentations.random_cutout, 'params': {'min_cut': 5, 'max_cut': 15}},
    {'func': augmentations.random_cutout, 'params': {'min_cut': 10, 'max_cut': 20}},
]


class RandomAugmentingEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Jumping env has 2 possible actions
        self.num_actions = 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(1, 60, 60), dtype=np.uint8)
        self.actualEnv = gym.make('jumping-task-v0')

    def step(self, action):
        idx = np.random.choice(range(len(POSSIBLE_AUGMENTATIONS)))
        augmentation = POSSIBLE_AUGMENTATIONS[idx]

        observation, r, done, info = self.actualEnv.step(action)
        # augment the observation
        aug_obs = augmentation['func'](observation, **augmentation['params'])
        # The augmented observation can have a different width and height!!
        # compensate for that

        return observation.astype('uint8').reshape(1, 60, 60), float(r), done, info

    def reset(self):
        observation = self.actualEnv.reset()
        return observation.astype('uint8').reshape(1, 60, 60)  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()
