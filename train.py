import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

from rl.common.buffer2 import RolloutBuffer
from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker, FigureLogger, WandBLogger
from rl.ppo.drac_ppo import DrACPPO
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO
from torch import optim
from env import VanillaEnv, AugmentingEnv, UCBAugmentingEnv, TRAIN_CONFIGURATIONS
from rl.common.utils import set_seed
import logging
import argparse

from validate import validate

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--environment", choices=["vanilla", "random", "UCB"], default="vanilla",
                    help="The environment to train on")
parser.add_argument("-c", "--configuration", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                    help="The environment configuration to train on")
parser.add_argument("-a", "--algorithm", choices=["PPO", "DRAC"], default="PPO",
                    help="The algorithm to train with")
parser.add_argument("-ne", "--episodes", default=30000, type=int,
                    help="Number of episodes to train")
parser.add_argument("-hs", "--hidden_size", default=256, type=int,
                    help="Hidden size of the actor and critic network")

parser.add_argument("-lr", "--learning_rate", default=0.001, type=float,
                    help="Learning rate for the optimizer")

parser.add_argument("-bs", "--batch_size", default=1000, type=int,
                    help="Size of the minibatch used for one gradient update")

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
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

logger1 = ConsoleLogger(log_every=1000, average_over=100)
logger2 = FigureLogger()
logger3 = TensorboardLogger('./tensorboard', run_id=run_name, info=vars(args))
logger4 = WandBLogger('practical work', info=vars(args))
tracker = Tracker(logger1, logger2, logger3, logger4)

buffer = RolloutBuffer(capacity=2000, batch_size=args.batch_size, min_transitions=2000)
if args.algorithm == "PPO":
    alg = PPO(policy, env, optimizer, seed=seed, tracker=tracker, buffer=buffer, n_epochs=10)
else:
    alg = DrACPPO(policy, env, optimizer, seed=31, tracker=tracker, buffer=buffer, n_epochs=10,
                  alpha_policy=args.alpha_policy, alpha_value=args.alpha_value)

logging.info("Training on " + alg.device)
alg.learn(args.episodes)
alg.save(run_path, str(args.episodes), info={'conf': conf})

fig = logger2.get_figure(fig_size=(8, 4))
fig.suptitle(f"Training {run_name} for {args.episodes} episodes")
plt.savefig(run_path + os.sep + 'training.png')

gp = validate(alg.policy, seed)
logging.info(f"Generalization performance (across all configurations): {gp:.4f}")
