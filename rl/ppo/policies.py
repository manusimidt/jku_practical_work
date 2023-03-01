import numpy as np
import torch
import torch.nn as nn

class PPOPolicy(nn.Module):
    def __init__(self, num_actions, obs_dim) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.obs_dim = obs_dim

class ActorNet(nn.Module):
    """Actor network (policy) """
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=True),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_size, action_size)
        self.fc_sigma = nn.Linear(hidden_size, action_size)




class ActorCriticCnnPolicy(PPOPolicy):
    def __init__(self, num_actions:int=2, obs_dim:int = 60) -> None:
        super().__init__(num_actions, obs_dim)