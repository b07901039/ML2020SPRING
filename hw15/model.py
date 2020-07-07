"""
Policy Gradient network & agent
Baseline network & agent
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyGradientNetwork(nn.Module):

    def __init__(self, hidden_size=16):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

class PolicyGradientAgent():

    def __init__(self, network, momentum=0, lr=1e-3, scheduler=None, optimizer="SGD"):
        self.network = network.cuda()
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.network.parameters(), lr=lr, momentum=momentum)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        else:
            raise NameError("Unknown optimizer type")

        if scheduler is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, scheduler["step_size"], scheduler["gamma"], last_epoch=-1)
        else:
            self.scheduler = None

    def learn(self, log_probs, rewards):

        loss = (-log_probs.double().cuda() * rewards.double().cuda()).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state).cuda())
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

class PolicyGradientNetwork_2(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = F.relu(self.fc1(state))
        hid = F.relu(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

class BaselineNetwork(nn.Module):

    def __init__(self, hidden_size=10):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, state):
        hid = F.relu(self.fc1(state))
        return self.fc2(hid).squeeze()

class BaselineAgent():

    def __init__(self, network):
        self.network = network.cuda()
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def learn(self, states, rewards):

        outputs = self.network(torch.Tensor(states).cuda())
        loss = self.criterion(outputs, torch.Tensor(rewards).cuda())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()