import gym
import stable_baselines3.common.env_checker
import gym_jumping_task
import numpy as np
from stable_baselines3 import DQN
from gym import spaces

num_actions = 2


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_actions)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(1, 60, 60), dtype=np.uint8)
        self.actualEnv = gym.make('jumping-task-v0')

    def step(self, action):
        observation, r, done, info = self.actualEnv.step(action)
        return observation.astype('uint8').reshape(1, 60, 60), float(r), done, info

    def reset(self):
        observation = self.actualEnv.reset()
        return (observation*255).astype('uint8').reshape(1, 60, 60)  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


env = CustomEnv()
stable_baselines3.common.env_checker.check_env(env)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard", buffer_size=100_000)
model.learn(total_timesteps=10000000)
