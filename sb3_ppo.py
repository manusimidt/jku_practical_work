from stable_baselines3 import PPO
from env import VanillaEnv, AugmentingEnv, UCBAugmentingEnv

env = VanillaEnv(
    [
        (29, 12), (31, 12), 
        (29, 20), (31, 20), 
    ]
)

env = UCBAugmentingEnv(c=200)
# policy = "MlpPolicy"
policy = "CnnPolicy"

model = PPO(policy, env, verbose=1, tensorboard_log="./tensorboard")
print(model.policy)
model.learn(total_timesteps=500_000, tb_log_name='SB3-UCBc200-' + policy)
print("current aug count", list(env.N))
