import gym
import stable_baselines3.common.env_checker
from stable_baselines3.common.callbacks import BaseCallback

import gym_jumping_task
import numpy as np
from stable_baselines3 import DQN, PPO
from gym import spaces

from env import RandomAugmentingEnv

num_actions = 2
         
env = RandomAugmentingEnv(obs_position=20, floor_height=20)


stable_baselines3.common.env_checker.check_env(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard")
model.learn(total_timesteps=10000000)
