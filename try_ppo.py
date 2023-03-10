import gym
import gym_jumping_task

from rl.ppo.policies import ActorCriticNet
from torch import optim

from rl.ppo.ppo import PPO

env = gym.make('jumping-task-v0')
num_actions = 2
num_episodes = 100

policy: ActorCriticNet = ActorCriticNet()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

ppo = PPO(policy, env, optimizer)
ppo.learn(10000000)
