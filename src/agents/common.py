import random
from collections import namedtuple

import torch
from torch import nn

from environment import Bid

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Got from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._random_state = random.Random(1988)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self._random_state.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TrainedPlayerNetwork(nn.Module):
    N_CARDS_PER_COLOR = 14
    N_TRUMPS_AND_EXCUSES = 22

    def __init__(self):
        super(TrainedPlayerNetwork, self).__init__()

        nn_width = 64
        self.standard_cards_tower = nn.Sequential(
            nn.Linear(TrainedPlayerNetwork.N_CARDS_PER_COLOR, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )
        self.trump_tower = nn.Sequential(
            nn.Linear(TrainedPlayerNetwork.N_TRUMPS_AND_EXCUSES, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )
        self.merge_tower = nn.Sequential(
            nn.BatchNorm1d(5 * nn_width),
            nn.Linear(5 * nn_width, 8 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, 8 * nn_width),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, len(list(Bid)))
        )

    def forward(self, x):
        n = TrainedPlayerNetwork.N_CARDS_PER_COLOR
        x_color_1 = self.standard_cards_tower(x[:, :n])
        x_color_2 = self.standard_cards_tower(x[:, n:2 * n])
        x_color_3 = self.standard_cards_tower(x[:, 2 * n:3 * n])
        x_color_4 = self.standard_cards_tower(x[:, 3 * n:4 * n])
        x_trumps = self.trump_tower(x[:, 4 * n:])
        x = torch.cat([x_color_1, x_color_2, x_color_3, x_color_4, x_trumps], dim=1)
        x = self.merge_tower(x)
        x = self.output_layer(x)
        return x
