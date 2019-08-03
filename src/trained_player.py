import math
from collections import namedtuple

import numpy as np
import torch
from torch import nn, tensor

from environment import Card, Bid, GamePhase


def bid_phase_observation_encoder(observation):
    return tensor([card in observation["hand"] for card in list(Card)]).float()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Got from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._random_state = np.random.RandomState(1988)

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


class BidPhaseAgent:

    def __init__(self, model):
        self._model = model
        self._steps_done = 0

        # Training parameters
        self._eps_start = 0.9
        self._eps_end = 0.05
        self._eps_decay = 200
        self._random_state = np.random.RandomState(1988)

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.BID:
            raise ValueError("Invalid game phase")

        state = bid_phase_observation_encoder(observation)

        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(
            -1. * self._steps_done / self._eps_decay)
        self._steps_done += 1
        if self._random_state.rand() > eps_threshold:
            with torch.no_grad():
                output = self._model(state).max(1)[1].view(1, 1)
        else:
            output = torch.argmax(torch.tensor([self._random_state.rand(self._model[-1].out_features)])).item()

        return Bid(output)

    @staticmethod
    def create_dqn():
        input_size = len(list(Card))
        nn_width = 128
        return nn.Sequential(
            nn.Linear(input_size, nn_width),
            nn.ReLU(),
            nn.Linear(nn_width, len(list(Bid)))
        )
