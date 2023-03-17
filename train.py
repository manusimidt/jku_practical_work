import random
import numpy as np
import torch
from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO
from torch import optim
from env import VanillaEnv, AugmentingEnv, UCBAugmentingEnv
from rl.common.utils import set_seed

train_conf = {
    "narrow_grid": {
        # (obstacle_pos, floor_height)
        (26, 12), (29, 12), (31, 12), (34, 12),
        (26, 20), (29, 20), (31, 20), (34, 20),
        (26, 28), (29, 28), (31, 28), (34, 28),
    },
    "wide_grid": {
        # (obstacle_pos, floor_height)
        (22, 8), (27, 8), (32, 8), (38, 8),
        (22, 20), (27, 20), (32, 20), (38, 20),
        (22, 32), (27, 32), (32, 32), (38, 32),
    },
    "random_grid": {
        # (obstacle_pos, floor_height)
        (15, 36), (17, 8), (19, 20), (21, 32),
        (26, 20), (30, 4), (32, 24), (34, 36),
        (36, 4), (38, 16), (43, 12), (44, 28),
    }
}
#environment = 'vanilla_env'
# environment = 'rnd_aug_env'
environment = 'UCB_aug_env'

seed = 31

for conf_name in list(train_conf.keys())[1:]:
    current_configurations = list(train_conf[conf_name])
    env = None
    if 'vanilla' in environment:
        env = VanillaEnv(configurations=current_configurations)
    elif 'rnd' in environment:
        env = AugmentingEnv(configurations=current_configurations)
    elif 'UCB' in environment:
        env = UCBAugmentingEnv(configurations=current_configurations, c=170)
    else:
        exit(-1)


    set_seed(env, seed)
    run_name = environment + '-' + conf_name
    print(f"====== Training {run_name} ======")

    policy: ActorCriticNet = ActorCriticNet()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    logger1 = ConsoleLogger(log_every=1000, average_over=100)
    logger2 = TensorboardLogger('./tensorboard3', run_id=run_name)
    tracker = Tracker(logger1, logger2)

    ppo = PPO(policy, env, optimizer, seed=seed, tracker=tracker)
    print("Training on ", ppo.device)
    ppo.learn(15_000)
    ppo.save('./ckpts', run_name + '-15000', info={'conf': list(train_conf[conf_name])})
    # ppo.learn(15_000)
    # ppo.save('./ckpts', run_name + '-30000', info={'conf': list(train_conf[conf_name])})
