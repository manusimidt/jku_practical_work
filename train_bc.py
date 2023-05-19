import numpy as np
from env import VanillaEnv, TRAIN_CONFIGURATIONS
from rl.common.buffer2 import Transition, Episode
from rl.ppo.policies import ActorCriticNet
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
from rl.ppo.ppo import PPO


class BCDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def generate_expert_trajectories(env, n_episodes):
    gamma = 0.99

    episodes: List[Episode] = []

    for i in range(n_episodes):
        done = False
        episode = Episode(discount=gamma)
        obs = env.reset()
        obstacle_position = env.actualEnv.obstacle_position
        jumping_pixel = obstacle_position - 14
        step = 0
        while not done:
            action = 0 if step < jumping_pixel else 1
            next_obs, reward, done, _ = env.step(action)
            episode.append(Transition(obs, action, reward, 0))
            obs = next_obs
            env.render()
            step += 1

        episodes.append(episode)

    # get the states, returns and actions out of the episodes
    states, actions, values = [], [], []
    for episode in episodes:
        states += episode.states()
        actions += episode.actions()
        values += episode.calculate_return()

    states, actions, values = np.array(states), np.array(actions), np.array(values)
    # vertically add actions and values
    targets = np.column_stack((actions, values))

    X = torch.tensor(states)
    Y = torch.tensor(targets)
    data: BCDataset = BCDataset(X, Y)
    train_set_length = int(len(data) * 0.8)
    train_set, val_set = torch.utils.data.random_split(data, [train_set_length, len(data) - train_set_length])

    trainloader: DataLoader = DataLoader(train_set, batch_size=64, shuffle=True)
    testloader: DataLoader = DataLoader(val_set, batch_size=64, shuffle=True)
    return trainloader, testloader


@torch.enable_grad()
def train(net: ActorCriticNet, dataLoader: DataLoader, optim_actor, optim_critic,
          loss_actor=nn.CrossEntropyLoss(), loss_critic=nn.MSELoss()) -> tuple:
    device = next(net.parameters()).device
    net.train()
    actor_errors, critic_errors = [], []

    for batch in dataLoader:
        # X is the observation
        # y contains the choosen action and the return estimate from the critic
        X, y = batch[0].to(device), batch[1].to(device)
        target_actions = y[:, 0]
        target_returns = y[:, 1]

        # normalize returns
        target_returns = (target_returns - target_returns.mean()) / (target_returns.std() + 1e-5)

        pred_action_logits = net.actor.forward(X)
        pred_values = net.critic.forward(X).squeeze()

        actor_error = loss_actor(pred_action_logits, target_actions.to(torch.int64))
        critic_error = loss_critic(pred_values, target_returns.to(torch.float32))

        optim_actor.zero_grad()
        actor_error.backward()
        optim_actor.step()

        optim_critic.zero_grad()
        critic_error.backward()
        optim_critic.step()

        actor_errors.append(actor_error.item())
        critic_errors.append(critic_error.item())

    return actor_errors, critic_errors


@torch.no_grad()
def evaluate(net: ActorCriticNet, dataLoader: DataLoader, loss_actor, loss_critic) -> tuple:
    device = next(net.parameters()).device
    net.eval()
    actor_errors, critic_errors = [], []

    for batch in dataLoader:
        X, y = batch[0].to(device), batch[1].to(device)
        target_actions = y[:, 0]
        target_returns = y[:, 1]

        # normalize returns
        target_returns = (target_returns - target_returns.mean()) / (target_returns.std() + 1e-5)

        pred_action_logits = net.actor.forward(X)
        pred_values = net.critic.forward(X).squeeze()

        actor_errors.append(loss_actor(pred_action_logits, target_actions.to(torch.int64)).item())
        critic_errors.append(loss_critic(pred_values, target_returns.to(torch.float32)).item())

    return actor_errors, critic_errors


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on ", device)

lr = 0.001
n_episodes = 10_000
n_epochs = 6

environments = ['vanilla_env', 'rnd_aug_env']
for environment in environments:
    for conf_name in list(TRAIN_CONFIGURATIONS.keys()):
        run_name = 'BC-' + environment + '-' + conf_name
        print(f"====== Training on {run_name} ======")

        env = VanillaEnv(configurations=list(TRAIN_CONFIGURATIONS[conf_name]), rendering=True)

        print(f"Generating {n_episodes} expert episodes...")
        trainloader, testloader = generate_expert_trajectories(env, n_episodes)

        model = ActorCriticNet(obs_space=(1, 60, 60), action_space=2, hidden_size=128).to(device)
        optim_actor = optim.Adam(model.actor.parameters(), lr=lr)
        optim_critic = optim.Adam(model.critic.parameters(), lr=lr)
        loss_actor = nn.CrossEntropyLoss()
        loss_critic = nn.MSELoss()

        for epoch in range(n_epochs):
            actor_errors, critic_errors = train(model, trainloader, optim_actor, optim_critic, loss_actor, loss_critic)
            val_actor_errors, val_critic_errors = evaluate(model, testloader, loss_actor, loss_critic)

            print(
                f"""Epoch {epoch} Train errors: Actor {np.mean(actor_errors):.3f}, Critic {np.mean(actor_errors):.3f} Val errors: Actor {np.mean(val_actor_errors):.3f}, Critic {np.mean(val_critic_errors):.3f} """)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        ppo = PPO(model, env, optimizer)

        ppo.save('./ckpts-final', run_name, info={'conf': list(TRAIN_CONFIGURATIONS[conf_name])})
