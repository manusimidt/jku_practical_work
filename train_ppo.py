from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO
from torch import optim
from env import VanillaEnv, AugmentingEnv, UCBAugmentingEnv

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
#environment = 'vanilla'
#environment = 'random'
environment = 'UCB'


for conf_name in train_conf.keys():
    current_configurations = list(train_conf[conf_name])
    env = None
    if 'vanilla' in environment:
        env = VanillaEnv(configurations=current_configurations)
    elif 'random' in environment:
        env = AugmentingEnv(configurations=current_configurations)
    else:
        env = UCBAugmentingEnv(configurations=current_configurations, c=170)

    run_name = environment + '-test' + conf_name
    print(f"====== Training {run_name} ======")

    policy: ActorCriticNet = ActorCriticNet(obs_space=(1, 60, 60), action_space=2, hidden_size=128)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    logger1 = ConsoleLogger(log_every=1000, average_over=100)
    logger2 = TensorboardLogger('./tensorboard2', run_id=run_name)
    tracker = Tracker(logger1, logger2)

    ppo = PPO(policy, env, optimizer, seed=31, tracker=tracker)
    print("Training on ", ppo.device)
    ppo.learn(15_000)
    ppo.save('./ckpts', run_name + '-15000', info={'conf': list(train_conf[conf_name])})
    ppo.learn(15_000)
    ppo.save('./ckpts', run_name + '-30000', info={'conf': list(train_conf[conf_name])})
