import gym
import gym_jumping_task
import numpy as np

env = gym.make('jumping-task-v0')
num_actions = 2

num_episodes = 100

state = env.reset()

for i in range(num_episodes):
    done = False
    rewards = []

    while not done:
        action = np.random.randint(low=0, high=num_actions)
        next_state, r, done, info = env.step(action)
        rewards.append(r)

        state = next_state
        if done: 
            print("Done! Episode Reward: ", np.sum(rewards))
            state = env.reset()
            break