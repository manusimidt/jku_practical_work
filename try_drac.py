from env import VanillaEnv, AugmentingEnv
from rl.common.logger import ConsoleLogger, Tracker, TensorboardLogger
from rl.dqn.dqn import DQN
from rl.dqn.policies import CNNDQNNet, DQNPolicy
from torch import optim
from rl.ppo.policies import ActorCriticNet

from rl.ppo.drac_ppo import DrACPPO

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
train_conf = {
    "temp": {(30, 10), }
}

for conf_name in train_conf.keys():
    print(f"====== Training on {conf_name} ======")
    current_configurations = list(train_conf[conf_name])
    env = AugmentingEnv(configurations=current_configurations)

    policy: ActorCriticNet = ActorCriticNet()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    logger1 = ConsoleLogger(log_every=1000, average_over=100)
    logger2 = TensorboardLogger('./tensorboard', run_id='drag-0-' + conf_name)
    tracker = Tracker(logger1, logger2)

    drac = DrACPPO(policy, env, optimizer, tracker=tracker, alpha=0.0)
    print("Training on ", drac.device)
    drac.learn(30_000)
    drac.save('./ckpts', 'drag-0' + conf_name, info={'conf': list(train_conf[conf_name])})