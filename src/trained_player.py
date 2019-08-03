import random
from collections import namedtuple

from torch import nn, tensor, argmax

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

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BidPhaseAgent:

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.BID:
            raise ValueError("Invalid game phase")

        state = bid_phase_observation_encoder(observation)

        nn_width = 128
        model = nn.Sequential(
            nn.Linear(state.shape[0], nn_width),
            nn.ReLU(),
            nn.Linear(nn_width, len(list(Bid)))
        )
        return Bid(argmax(model(state)).item())
