import gym
import gym_jumping_task
from gym_jumping_task.envs.jumping_task import JumpTaskEnv
import numpy as np
from rl.dqn.dqn import DQN
from rl.dqn.policies import CNNDQNNet, DQNPolicy
from torch import optim

class CustomEnv(gym.Env):
    """Custom Environment that follows"""
    def __init__(self):
        super().__init__()
        self.actualEnv = JumpTaskEnv(scr_w=60, scr_h=60)

    def step(self, action):
        observation, r, done, info = self.actualEnv.step(action)
        return observation.astype('uint8').reshape(1, 60, 60), float(r), done, info

    def reset(self):
        observation = self.actualEnv.reset()
        return observation.astype('uint8').reshape(1, 60, 60)  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()




num_actions = 2
num_episodes = 100


policy: DQNPolicy = CNNDQNNet(num_actions)

optimizer = optim.Adam(policy.parameters(), lr=0.001)

dqn = DQN(policy, env, optimizer)

dqn.learn(10000000)