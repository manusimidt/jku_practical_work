import gym
import torch
import numpy as np

from rl.common.buffer2 import Episode, Transition, RolloutBuffer
from rl.ppo.policies import ActorCriticNet


class PPO():
    def __init__(self,
                 policy: ActorCriticNet,
                 env: gym.Env,
                 optimizer: torch.optim.Optimizer,
                 metric=torch.nn.MSELoss(),
                 buffer: RolloutBuffer = RolloutBuffer(),
                 gamma=0.99,
                 eps_clip=0.2,
                 value_loss_scale: float = 0.5,
                 policy_loss_scale: float = 1.0,
                 entropy_loss_scale: float = 0.01,
                 use_buffer_reset=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ) -> None:
        """
        :param gamma: discount rate
        """
        self.policy = policy
        self.env = env
        self.optimizer = optimizer
        self.metric = metric
        self.buffer = buffer

        self.n_epochs = 4
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_loss_scale = value_loss_scale
        self.policy_loss_scale = policy_loss_scale
        self.entropy_loss_scale = entropy_loss_scale
        self.use_buffer_reset = use_buffer_reset
        self.device = device

    def train(self) -> None:
        """
        Update the policy using the currently gathered rollout buffer
        """
        self.policy.train()  # Switch to train mode (as apposed to eval)
        self.optimizer.zero_grad()

        for _ in range(self.n_epochs):
            # sample batch from buffer
            samples = self.buffer.sample()
            episode_returns = []
            states = []
            actions = []
            old_log_probs = []
            for s in samples:
                states += [s.state, ]  # todo check if this has to be transformed torch.FloatTensor(state).to(device)
                episode_returns += [s.g_return, ]
                actions += [s.action, ]
                old_log_probs += [s.log_probs, ]

            # create tensors
            state_t = torch.stack(states, dim=0).to(self.device).detach()
            action_t = torch.LongTensor(actions).to(self.device).detach()
            return_t = torch.FloatTensor(episode_returns).view(-1, 1).to(self.device)
            old_log_probs_t = torch.stack(old_log_probs, dim=0).to(self.device).detach()

            # normalize returns
            return_t = (return_t - return_t.mean()) / (return_t.std() + 1e-5)

            # get value function estimate
            v_s, log_probs, entropy = self.policy.evaluate(state_t, action_t)

            # use importance sampling
            ratios = torch.exp(log_probs.view(-1, 1) - old_log_probs_t.detach())

            # compute advantage
            advantages = return_t - v_s.detach()

            # compute ppo trust region loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2)

            # compute critic loss
            value_loss = self.metric(v_s, return_t)

            # combine losses
            loss = self.value_loss_scale * value_loss + \
                   self.policy_loss_scale * policy_loss - \
                   self.entropy_loss_scale * entropy.view(-1, 1)

            # perform training step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def learn(self, n_episodes) -> None:
        """
        Collect rollouts
        """
        state = self.env.reset()

        for i in range(n_episodes):
            # create a new episode
            episode = Episode(discount=self.gamma)

            done = False
            while not done:
                # select an action from the agent's policy
                # todo make sure the state is in the correct format!
                action, log_probs = self.policy.act(state)

                # enter action into the env
                next_state, reward, done, _ = self.env.step(action.item())
                episode.total_reward += reward

                # store agent trajectory
                transition = Transition(state=state, action=action, reward=(reward * self.reward_scale), log_probs=log_probs)
                episode.append(transition)

                # update agent if done
                if done:
                    # add current episode to the replay buffer
                    self.buffer.add(episode)

                    # skip if stored episodes are less than the batch size
                    if len(self.buffer) < self.buffer.min_transitions: break

                    # update the network
                    self.train()
                    self.buffer.update_stats()
                    if self.use_buffer_reset: self.buffer.reset()