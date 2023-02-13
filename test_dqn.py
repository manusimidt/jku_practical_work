import gym
import gym_jumping_task
import numpy as np
from rl.dqn.dqn import DQN
from rl.dqn.policies import CNNDQNNet, DQNPolicy
from torch import optim

env = gym.make('jumping-task-v0')
num_actions = 2
num_episodes = 100


policy: DQNPolicy = CNNDQNNet(num_actions)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

dqn = DQN(policy, env, optimizer)

dqn.learn(10000000)