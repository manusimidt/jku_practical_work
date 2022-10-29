from pickletools import optimize
import gym
import torch
import numpy as np
from copy import copy
from rl.dqn.policies import DQNPolicy
from rl.common.buffer import ReplayBuffer
from rl.common.utils import soft_update

class DQN():

    def __init__(self, 
        policy:DQNPolicy, 
        env:gym.Env, 
        optimizer: torch.optim.Optimizer,

        metric = torch.nn.MSELoss,
        buffer: ReplayBuffer = ReplayBuffer(), 
        minibatch_size = 128, 
        update_after = 2000,
        gamma = 0.99, 
        epsilon:float = 0.99 
    ) -> None:

        self.policy = policy
        self.dqn_target = copy(policy)
        self.env = env
        self.optimizer = optimizer
        self.metric = metric
        self.buffer = buffer
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.epsilon = epsilon

    def train(self, ):
        self.policy.train()
        self.optimizer.zero_grad()

        # Sample minibatch from replay buffer
        mini_batch = self.buffer.sample_batch(self.minibatch_size)
        states, actions, rewards, nextStates, dones = mini_batch

        # Compute q values for states
        target_q_value = self.dqn_target.forward(nextStates)

        # Compute the targets for training
        targets = rewards + (self.gamma * torch.max(target_q_value, 1).values * (1 - dones))

        # compute the predictions for training
        online_q_values = self.policy.forward(states)
        action_idx = actions.argmax(axis=1) 
        predictions = online_q_values.gather(1, action_idx.unsqueeze(1)).flatten()

        # Update the loss
        loss = self.metric(predictions, targets)
        loss.backwards(retain_graph=False)
        self.optimizer.step()
        
        soft_update(self.policy, self.dqn_target, self.tau)



    def learn(self, num_epochs:int):

        state = self.env.reset()

        for i in range(num_epochs):
            episode_return = 0
            done = False
            while not done:
                
                # epsilon decay
                epsilon = epsilon

                # epsilon greedy action selection
                if np.random.choice([True,False], p=[epsilon,1-epsilon]):
                    action = np.random.randint(low=0, high=self.policy.num_actions)
                else:
                    logits = self.policy.forward(state)
                    action = torch.argmax()

                # interact with the environment
                next_state, r, done, info = self.env.step(action)
                episode_return += r

                # update policy using temporal difference
                if self.buffer.length() > self.minibatch_size and self.buffer.length() > self.update_after:
                    self.train()
                
                if done:
                    state = self.env.reset()
                    print(f"Episode: \t{i}\t{episode_return}")
                    break