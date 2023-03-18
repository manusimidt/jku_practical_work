import matplotlib.pyplot as plt

from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker, FigureLogger
from rl.ppo.policies import ActorCriticNet
from rl.ppo.drac_ppo import DrACPPO
from torch import optim
from env import AugmentingEnv, UCBAugmentingEnv
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
    },
    "diagonal_grid": {
        (17, 8), (21, 12), (25, 16), (29, 20),
        (32, 24), (36, 28), (40, 32), (44, 36),
    },
}
# environment = 'rnd_aug_env'
# environment = 'UCB_aug_env'
environments = ['rnd_aug_env', 'UCB_aug_env']

seed = 31
for environment in environments:
    for conf_name in train_conf.keys():
        current_configurations = list(train_conf[conf_name])
        env = None

        if 'rnd' in environment:
            env = AugmentingEnv(configurations=current_configurations)
        elif 'UCB' in environment:
            env = UCBAugmentingEnv(configurations=current_configurations, c=200)
        else:
            exit(-1)

        set_seed(env, seed)
        run_name = 'DRAC-' + environment + '-' + conf_name
        print(f"====== Training {run_name} ======")

        policy: ActorCriticNet = ActorCriticNet()
        optimizer = optim.Adam(policy.parameters(), lr=0.001)

        logger1 = ConsoleLogger(log_every=1000, average_over=100)
        logger2 = TensorboardLogger('./tensorboard-final', run_id=run_name)
        logger3 = FigureLogger()
        tracker = Tracker(logger1, logger2, logger3)

        drac = DrACPPO(policy, env, optimizer, seed=31, tracker=tracker, alpha_policy=0.2, alpha_value=0.05)
        print("Training on ", drac.device)
        drac.learn(15)
        drac.save('./ckpts-final', run_name + '-15000', info={'conf': list(train_conf[conf_name])})
        drac.learn(15)
        drac.save('./ckpts-final', run_name + '-30000', info={'conf': list(train_conf[conf_name])})

        fig = logger3.get_figure(fig_size=(8, 4))
        fig.suptitle(f"Training {run_name} for 30000 episodes")
        plt.savefig('./ckpts-final/' + run_name + '.png')
