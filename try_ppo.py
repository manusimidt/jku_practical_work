from rl.ppo.policies import ActorCriticNet
from torch import optim
from env import VanillaEnv
from rl.ppo.ppo import PPO

# env = VanillaEnv(obs_positions=(30, 15), floor_heights=(20, 15, 30))
env = VanillaEnv()
num_actions = 2
num_episodes = 100

policy: ActorCriticNet = ActorCriticNet()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

ppo = PPO(policy, env, optimizer)
ppo.learn(10000000)
