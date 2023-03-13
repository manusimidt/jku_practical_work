from env import VanillaEnv
from rl.dqn.dqn import DQN
from rl.dqn.policies import CNNDQNNet, DQNPolicy
from torch import optim

env = VanillaEnv()



policy: DQNPolicy = CNNDQNNet()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

dqn = DQN(policy, env, optimizer)

dqn.learn(10000000)