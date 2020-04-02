import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F


# def hidden_init(input_layer):
#     """
#     Using Xavier/Glorot Initialization because it has been
#     proven that it is better with using tanh activation function
#     """
#
#     fan_in = input_layer.weight.data.size()[0]
#     lim = 2 / np.sqrt(fan_in)
#     return -lim, lim


# def init_weight(layer):
#     """
#         Initializing the weights of the network
#         using uniform
#         """
#     torch.nn.init.uniform_(layer, *hidden_init(layer))


class Actor(nn.Module):
    """Actor (Value) Model."""

    def __init__(self, state_size, action_size, tau=0.001):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()
        self.tau = tau

        self.in_fn = torch.nn.BatchNorm1d(state_size)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        self.fc1 = torch.nn.Linear(state_size, 128)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.fc2 = torch.nn.Linear(128, 128)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.fc3 = torch.nn.Linear(128, action_size)

        self.fc3.weight.data.uniform_(-0.003, 0.003)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        fc1 = self.relu1(self.fc1(self.in_fn(inputs)))
        fc2 = self.relu2(self.fc2(fc1))

        fc3 = self.fc3(fc2)

        return fc3

    def train_step(self, critic, states, actions, curr_pol_out):
        actor_loss = -critic(states, actions).mean()
        actor_loss += (curr_pol_out ** 2).mean() * 1e-3

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

    def hard_copy(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(param.data)


class Critic(nn.Module):
    # def init_weights(self, m):
    #     I.xavier_uniform_(m.weight, gain=1)

    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, tau=0.001):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()

        self.tau = tau
        self.in_fn = torch.nn.BatchNorm1d(state_size + action_size)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        self.fc1 = torch.nn.Linear(state_size + action_size, 128)

        self.fc2 = torch.nn.Linear(128, 128)

        self.fc3 = torch.nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.ReLU = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs, actions):
        h1 = self.ReLU(self.fc1(self.in_fn(torch.cat([inputs, actions], dim=1))))
        h2 = self.ReLU(self.fc2(h1))
        Q = self.fc3(h2)

        return Q

    def train_step(self, states, actions, yi):
        current_Q = self(states, actions)

        critic_loss = F.mse_loss(current_Q, yi)

        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

    def hard_copy(self, critic):
        for param, target_param in zip(critic.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
