"""
This module holds different modified jumping tasks environments
All environments should return uint8 observations in range [0..255] because stable baselines
expects image input to be in that range.
"""

import gym
import numpy as np
import torch
from gym import spaces
from typing import List
import itertools
import augmentations
from gym_jumping_task.envs import JumpTaskEnv
import torchvision.transforms.functional as fn

POSSIBLE_AUGMENTATIONS = [
    {'func': augmentations.identity, 'params': {}},
    # {'func': augmentations.identity, 'params': {}},
    # todo probably best to shift the image just by a few pixels and not 10, 20 and 30
    # See learning invariances for policy generalization Paper
    {'func': augmentations.random_translate, 'params': {'size': 64}},
    {'func': augmentations.random_translate, 'params': {'size': 68}},
    {'func': augmentations.random_translate, 'params': {'size': 72}},
     #{'func': augmentations.random_crop, 'params': {'out': 55}},
     #{'func': augmentations.random_crop, 'params': {'out': 50}},
     #{'func': augmentations.random_crop, 'params': {'out': 45}},
     #{'func': augmentations.random_cutout, 'params': {'min_cut': 2, 'max_cut': 5}},
     #{'func': augmentations.random_cutout, 'params': {'min_cut': 5, 'max_cut': 15}},
     #{'func': augmentations.random_cutout, 'params': {'min_cut': 10, 'max_cut': 20}},
]


class VanillaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    min_obstacle_pos = 14
    max_obstacle_pos = 47
    min_floor_height = 0
    max_floor_height = 40

    def __init__(self, configurations: List[tuple] or None = None, rendering=False):
        """
        :param configurations: possible configurations, array of tuples consisting of
            the obstacle position and the floor height
        """
        super().__init__()
        # If no configuration was provided, use the default JumpingTask configuration
        if configurations is None: configurations = [(30, 10), ]
        self.configurations = configurations

        # Jumping env has 2 possible actions
        self.num_actions = 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 60, 60), dtype=np.uint8)
        conf = self._sample_conf()
        self.actualEnv = JumpTaskEnv(obstacle_position=conf[0], floor_height=conf[1], rendering=rendering)

    def _sample_conf(self):
        """
        :return: returns random configuration as a tuple consisting of the obstacle position and
            the floor height
        """
        idx = np.random.choice(len(self.configurations))
        return self.configurations[idx]

    def step(self, action) -> tuple:
        obs, r, done, info = self.actualEnv.step(action)
        return (obs * 255).astype('uint8').reshape(1, 60, 60), float(r), done, info

    def reset(self) -> np.ndarray:
        conf = self._sample_conf()
        obs = self.actualEnv._reset(obstacle_position=conf[0], floor_height=conf[1])
        return (obs * 255).astype('uint8').reshape(1, 60, 60)

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


class RandomAugmentingEnv(VanillaEnv):
    """Custom Environment that follows gym interface."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, configurations: List[tuple] or None = None, rendering=False):
        """
        :param configurations: possible configurations, array of tuples consisting of
            the obstacle position and the floor height
        """
        super().__init__(configurations, rendering)
        self.current_augmentation = None
        self._sample_augmentation()

    def step(self, action):
        obs, r, done, info = super().step(action)
        return self._augment(obs), r, done, info

    def reset(self):
        obs = super().reset()
        aug_obs = self._augment(obs)
        # sample a new augmentation
        self._sample_augmentation()
        return aug_obs

    def _sample_augmentation(self):
        idx = np.random.choice(range(len(POSSIBLE_AUGMENTATIONS)))
        self.current_augmentation = POSSIBLE_AUGMENTATIONS[idx]

    def _augment(self, obs):
        augmentation = self.current_augmentation
        # convert the observation in the needed format (B x C x H x W) [0..255] int8
        aug_obs = np.expand_dims(obs, axis=1)
        # augment the observation
        aug_obs = augmentation['func'](aug_obs, **augmentation['params'])
        aug_obs = aug_obs.squeeze(axis=0)
        # The augmented observation can have a different width and height!!
        # compensate for that
        if not obs.shape == aug_obs.shape:
            aug_obs = fn.resize(torch.from_numpy(aug_obs), size=[60, 60]).numpy()
        return aug_obs


class UCBAugmentingEnv(RandomAugmentingEnv):
    def __init__(self, configurations: List[tuple] or None = None, rendering=False, c=2):
        """
        :param configurations: possible configurations, array of tuples consisting of
            the obstacle position and the floor height
        """
        # UCB Exploration coefficient
        self.c = c
        # N_t number of times each transformation was selected
        self.N = np.zeros(len(POSSIBLE_AUGMENTATIONS))
        # total number of selections
        self.t = 0
        # UCB Q function
        self.Q = np.zeros(len(POSSIBLE_AUGMENTATIONS))
        self.episode_return = 0
        self.curr_aug_idx = None
        super().__init__(configurations, rendering)

    def _sample_augmentation(self):
        # UCB: If N_t(aug_idx) = 0, choose this action!
        if 0 in self.N:
            self.curr_aug_idx = np.where(self.N == 0)[0][0]
        else:
            self.curr_aug_idx = np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))
        self.current_augmentation = POSSIBLE_AUGMENTATIONS[self.curr_aug_idx]

    def step(self, action):
        obs, r, done, info = super().step(action)
        # collect the return for UCB calculation
        self.episode_return += r
        return obs, r, done, info

    def reset(self):
        # Update Q
        curr_aug = self.curr_aug_idx
        # Make sure that this is not the first reset called before a step
        if curr_aug is not None:
            self.N[curr_aug] += 1
            self.Q[curr_aug] = self.Q[curr_aug] + 1 / self.N[curr_aug] * (self.episode_return - self.Q[curr_aug])
            self.t += 1
            # reset episode return
            self.episode_return = 0
        return super().reset()


if __name__ == '__main__':
    import stable_baselines3.common.env_checker as env_checker

    _envs = [VanillaEnv(), RandomAugmentingEnv(), UCBAugmentingEnv()]
    for _env in _envs:
        env_checker.check_env(_env)
        _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
        for _obs in _obs_arr:
            assert _obs.dtype == np.uint8, "Incorrect datatype"
            assert _obs.shape == (1, 60, 60), "Incorrect shape"
            assert 0 in _obs, "No grey pixels present"
            assert 127 in _obs, "No white pixels present"
            assert 255 in _obs, "No black pixels present"
