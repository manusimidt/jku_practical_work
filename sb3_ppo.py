import stable_baselines3.common.env_checker
from stable_baselines3 import PPO
from env import VanillaEnv, RandomAugmentingEnv, UCBAugmentingEnv

env = RandomAugmentingEnv(configurations=[
        (22, 8), (27, 8), (32, 8), (38, 8),
        (22, 20), (27, 20), (32, 20), (38, 20),
        (22, 32), (27, 32), (32, 32), (38, 32),
])

stable_baselines3.common.env_checker.check_env(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard")
model.learn(total_timesteps=1_000_000, tb_log_name="wide-shift")
model.save('wide-shift')
print("======= CHOSEN AUGMENTATION COUNTS =======")
# print(env.N)
print("======= CHOSEN AUGMENTATION COUNTS =======")
