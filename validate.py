import numpy as np
import torch

from env import VanillaEnv
from rl.common.logger import Logger
from rl.common.utils import set_seed
from rl.ppo.policies import ActorCriticNet

n_obstacle_pos = 26  # how many obstacle position you want to try out (paper: 27, max: 30)
n_floor_heights = 11  # how many floor heights you want to try out (paper: 11, max: 40)
obstacle_pos = np.rint(np.linspace(VanillaEnv.min_obstacle_pos, VanillaEnv.max_obstacle_pos, n_obstacle_pos)).astype(
    np.int8)
floor_height = np.rint(np.linspace(VanillaEnv.min_floor_height, VanillaEnv.max_floor_height, n_floor_heights)).astype(
    np.int8)


def validate(policy: ActorCriticNet, seed: int = 31) -> float:
    """
    Takes the model, evaluates it on all possible environment configurations and returns the generalization performance
    """
    solved_counter, failed_counter = 0, 0
    for obs_pos in obstacle_pos:
        for floor_h in floor_height:
            env = VanillaEnv([(obs_pos, floor_h), ])
            set_seed(env, seed)
            done = False
            obs = env.reset()
            while not done:
                action, _ = policy.act_deterministic(torch.FloatTensor(obs).unsqueeze(0))
                obs, _, done, info = env.step(action.item())
            if info['collision']:
                failed_counter += 1
            else:
                solved_counter += 1
    return solved_counter / (solved_counter + failed_counter) * 100


if __name__ == '__main__':
    import time

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _policy: ActorCriticNet = ActorCriticNet(obs_space=(1, 60, 60), action_space=2, hidden_size=256).to(_device)
    _ckp: dict = torch.load('./runs/2023-05-19-153945-PPO-random-wide_grid/15000.pth', map_location=_device)
    _policy.load_state_dict(_ckp['state_dict'])

    start = time.time()
    print("performance (%):", validate(_policy, 31))
    end = time.time()
    print("run time (sec):", end - start)
