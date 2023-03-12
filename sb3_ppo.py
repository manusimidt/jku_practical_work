import stable_baselines3.common.env_checker
from stable_baselines3 import PPO
from env import VanillaEnv

env = VanillaEnv(configurations=[
    (26, 12), (29, 12), (31, 12), (34, 12),
    (26, 20), (29, 20), (31, 20), (34, 20),
    (26, 28), (29, 28), (31, 28), (34, 28),
])

stable_baselines3.common.env_checker.check_env(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard")
model.learn(total_timesteps=1_000_000)
