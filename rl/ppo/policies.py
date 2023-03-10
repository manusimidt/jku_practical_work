import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,  stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,  stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*3*3, 128), 
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        # convert the images to a matrix with the batch count as first dimension and the features as second dimension
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # return torch.softmax(x, dim=-1) #-1 to take softmax of last dimension
        return x
    
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,  stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,  stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*3*3, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    
class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.actor = ActorNet()
        self.critic = CriticNet()

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action, log_probs
    
    def evaluate(self, state, action):
        action_logits = self.actor(state)
        policy_dist = Categorical(logits=action_logits)
        log_probs = policy_dist.log_prob(action)
        entropy = policy_dist.entropy()
        state_value = self.critic(state)
        return state_value, log_probs, entropy