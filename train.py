import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

from rl.common.buffer2 import RolloutBuffer
from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker, FigureLogger
from rl.ppo.drac_ppo import DrACPPO
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO
from torch import optim
from env import VanillaEnv, AugmentingEnv, UCBAugmentingEnv, TRAIN_CONFIGURATIONS
from rl.common.utils import set_seed
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--environment", choices=["vanilla", "random", "UCB"], default="vanilla",
                    help="The environment to train on")
parser.add_argument("-c", "--configuration", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                    help="The environment configuration to train on")
parser.add_argument("-a", "--algorithm", choices=["PPO", "DRAC"], default="PPO",
                    help="The algorithm to train with")
parser.add_argument("-ne", "--episodes", default=30000, type=int,
                    help="Number of episodes to train")
parser.add_argument("-hs", "--hidden_size", default=128, type=int,
                    help="Hidden size of the actor and critic network")
parser.add_argument("-ap", "--alpha_policy", default=0.4, type=float,
                    help="DrAC alpha weight for policy regularization loss")
parser.add_argument("-av", "--alpha_value", default=0.2, type=float,
                    help="DrAC alpha weight for value regularization loss")

args = parser.parse_args()
run_dir = "./runs"
run_name = '-'.join(
    [datetime.today().strftime('%Y-%m-%d-%H%M%S'), args.algorithm, args.environment, args.configuration])
run_path = run_dir + os.sep + run_name
os.makedirs(run_path)

# set up the logging to log both to STDOUT and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(run_path + os.sep + "console.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

seed = 31
episodes = 30000

conf = list(TRAIN_CONFIGURATIONS[args.configuration])
env = None
if 'vanilla' in args.environment:
    env = VanillaEnv(configurations=conf)
elif 'random' in args.environment:
    env = AugmentingEnv(configurations=conf)
elif 'UCB' in args.environment:
    env = UCBAugmentingEnv(configurations=conf, c=200)

set_seed(env, seed)

print(f"Started Training, logfile: {run_name}.log")

policy: ActorCriticNet = ActorCriticNet(obs_space=(1, 60, 60), action_space=2, hidden_size=args.hidden_size)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

logger1 = ConsoleLogger(log_every=1000, average_over=100)
logger2 = TensorboardLogger('./tensorboard', run_id=run_name, info=vars(args))
logger3 = FigureLogger()
tracker = Tracker(logger1, logger2, logger3)

buffer = RolloutBuffer(capacity=10_000, batch_size=1024, min_transitions=10_000)
if args.algorithm == "PPO":
    alg = PPO(policy, env, optimizer, seed=seed, tracker=tracker, n_epochs=10)
else:
    alg = DrACPPO(policy, env, optimizer, seed=31, tracker=tracker, alpha_policy=args.alpha_policy,
                  alpha_value=args.alpha_value)

logging.info("Training on " + alg.device)
alg.learn(episodes)
alg.save(run_path, str(episodes), info={'conf': conf})

fig = logger3.get_figure(fig_size=(8, 4))
fig.suptitle(f"Training {run_name} for {episodes} episodes")
plt.savefig(run_path + os.sep + 'training.png')
