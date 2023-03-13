from stable_baselines3 import PPO
from env import VanillaEnv, AugmentingEnv, UCBAugmentingEnv

env = VanillaEnv(
    [
        (29, 12), (31, 12), 
        (29, 20), (31, 20), 
    ]
)

env = VanillaEnv()

policy = "MlpPolicy"
# policy = "CnnPolicy"

model = PPO(policy, env, verbose=1, tensorboard_log="./tensorboard")
print(model.policy)
model.learn(total_timesteps=1_000_000, tb_log_name='PPO-vanilla-' + policy)
