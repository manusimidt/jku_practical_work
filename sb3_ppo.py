import gym
import stable_baselines3.common.env_checker
from stable_baselines3.common.callbacks import BaseCallback

import gym_jumping_task
import numpy as np
from stable_baselines3 import DQN, PPO
from gym import spaces

from env import RandomAugmentingEnv, VanillaEnv

num_actions = 2
         
env = VanillaEnv(configurations=[(22, 18), (22, 24), (26, 18), (26, 24)])


stable_baselines3.common.env_checker.check_env(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard")
model.learn(total_timesteps=10000000)
