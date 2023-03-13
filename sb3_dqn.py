from stable_baselines3 import DQN

from env import AugmentingEnv, VanillaEnv

env = VanillaEnv(
    [
        (29, 12), (31, 12),
        (29, 20), (31, 20),
    ]
)

policy = "MlpPolicy"
# policy = "CnnPolicy"

model = DQN(policy, env, verbose=1, tensorboard_log="./tensorboard", buffer_size=100_000)
print(model.policy)
model.learn(total_timesteps=3_000_000, tb_log_name='DQN-vanilla-' + policy)
