"""
This module holds different modified jumping tasks environments
All environments should return uint8 observations in range [0..255] because stable baselines
expects image input to be in that range.
"""

import gym
import numpy as np
from gym import spaces
import itertools
import stable_baselines3.common.env_checker as env_checker
import augmentations
from gym_jumping_task.envs import JumpTaskEnv

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

# obstacle_pos: min: 14, max: 47
obstacle_positions = np.array(range(14, 48, 2))
# floor_height: min: 0, max: 40
floor_heights = np.array(range(0, 41, 3))
ALL_CONFIGURATIONS = set(itertools.product(obstacle_positions, floor_heights))

# obstacle_pos: min: 14, max: 47
# floor_height: min: 0, max: 40
TRAINING_CONFIGURATIONS = set([
    # (obstacle_pos, floor_height)
    (22, 18), (22, 24),
    (26, 18), (26, 24),
])

TEST_CONFIGURATIONS = ALL_CONFIGURATIONS - TRAINING_CONFIGURATIONS


class VanillaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, obs_position=30, floor_height=10):
        super().__init__()
        self.obs_position = obs_position
        self.floor_height = floor_height

        # Jumping env has 2 possible actions
        self.num_actions = 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 60, 60), dtype=np.uint8)
        self.actualEnv = JumpTaskEnv(obstacle_position=obs_position, floor_height=floor_height)

    def step(self, action) -> tuple:
        obs, r, done, info = self.actualEnv.step(action)
        return (obs * 255).astype('uint8').reshape(1, 60, 60), float(r), done, info

    def reset(self) -> np.ndarray:
        obs = self.actualEnv._reset(floor_height=self.floor_height, obstacle_position=self.obs_position)
        return (obs * 255).astype('uint8').reshape(1, 60, 60)

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


class RandomAugmentingEnv(VanillaEnv):
    """Custom Environment that follows gym interface."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, obs_position=30, floor_height=10):
        super().__init__(obs_position, floor_height)

    def step(self, action):
        idx = np.random.choice(range(len(POSSIBLE_AUGMENTATIONS)))
        augmentation = POSSIBLE_AUGMENTATIONS[idx]

        obs, r, done, info = self.actualEnv.step(action)
        # convert the observation in the needed format (B x C x H x W) [0..255] int8
        # observation = np.expand_dims(np.array([observation * 255], dtype=np.uint8), axis=1)
        # augment the observation
        # aug_obs = augmentation['func'](observation, **augmentation['params'])
        # The augmented observation can have a different width and height!!
        # compensate for that

        return (obs * 255).astype('uint8').reshape(1, 60, 60), float(r), done, info

    def reset(self):
        obs = self.actualEnv._reset(floor_height=self.floor_height, obstacle_position=self.obs_position)
        # TODO augment observation!
        return (obs * 255).astype('uint8').reshape(1, 60, 60)  # reward, done, info can't be included


if __name__ == '__main__':
    _envs = [VanillaEnv(), RandomAugmentingEnv()]
    for _env in _envs:
        env_checker.check_env(_env)
        _obs_arr = [_env.reset(), _env.step(0)[0]]
        for _obs in _obs_arr:
            assert _obs.dtype == np.uint8, "Incorrect datatype"
            assert _obs.shape == (1, 60, 60), "Incorrect shape"
            assert 0 in _obs, "No grey pixels present"
            assert 127 in _obs, "No white pixels present"
            assert 255 in _obs, "No black pixels present"
