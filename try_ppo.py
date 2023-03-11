from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO
from torch import optim
from env import VanillaEnv

possible_configurations = [(30, 10), (28, 10), (32, 10), (26, 10), (34, 10),
                           (30, 14), (28, 14), (32, 14), (26, 14), (34, 14)]

for i in range(10, len(possible_configurations)+1):
    current_configurations = possible_configurations[:i]
    print(current_configurations)
    env = VanillaEnv(configurations=current_configurations)

    num_actions = 2
    num_episodes = 100

    policy: ActorCriticNet = ActorCriticNet()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    logger1 = ConsoleLogger(log_every=300)
    logger2 = TensorboardLogger('./tensorboard', run_id='conf_' + str(len(current_configurations)))
    tracker = Tracker(logger1, logger2)

    ppo = PPO(policy, env, optimizer, seed=31, tracker=tracker)
    ppo.learn(15_000)
