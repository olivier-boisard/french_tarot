import random
from abc import abstractmethod, ABC
from collections import namedtuple
from typing import List

import numpy as np
import torch
from torch import nn, optim, tensor
from torch.utils.tensorboard import SummaryWriter

from environment import Card, CARDS

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Got from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._random_state = random.Random(1988)

    def push(self, state: torch.Tensor, action: int, next_state: torch.Tensor, reward: float):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return self._random_state.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class BaseCardNeuralNet(nn.Module):
    N_CARDS_PER_COLOR = 14
    N_TRUMPS_AND_EXCUSES = 22

    def __init__(self):
        super(BaseCardNeuralNet, self).__init__()

        nn_width = 64
        self.standard_cards_tower = nn.Sequential(
            nn.Linear(BaseCardNeuralNet.N_CARDS_PER_COLOR, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )
        self.trump_tower = nn.Sequential(
            nn.Linear(BaseCardNeuralNet.N_TRUMPS_AND_EXCUSES, nn_width),
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

    @property
    def output_dimensions(self) -> int:
        # noinspection PyUnresolvedReferences
        return self.merge_tower[-2].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = BaseCardNeuralNet.N_CARDS_PER_COLOR
        x_color_1 = self.standard_cards_tower(x[:, :n])
        x_color_2 = self.standard_cards_tower(x[:, n:2 * n])
        x_color_3 = self.standard_cards_tower(x[:, 2 * n:3 * n])
        x_color_4 = self.standard_cards_tower(x[:, 3 * n:4 * n])
        x_trumps = self.trump_tower(x[:, 4 * n:])
        x = torch.cat([x_color_1, x_color_2, x_color_3, x_color_4, x_trumps], dim=1)
        x = self.merge_tower(x)
        return x


class Agent(ABC):

    @abstractmethod
    def get_action(self, observation: dict):
        pass

    @abstractmethod
    def optimize_model(self, tb_writer: SummaryWriter):
        pass


class BaseNeuralNetAgent(Agent):
    def __init__(
            self,
            policy_net: nn.Module,
            eps_start: float = 0.9,
            eps_end: float = 0.05, eps_decay: int = 500, batch_size: int = 64,
            replay_memory_size: int = 2000
    ):
        self._policy_net = policy_net
        self._steps_done = 0
        self._random_state = np.random.RandomState(1988)

        # Training parameters
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._batch_size = batch_size
        self.memory = ReplayMemory(replay_memory_size)
        self._optimizer = optim.Adam(self._policy_net.parameters())

        self.loss = []

    @abstractmethod
    def optimize_model(self, tb_writer: SummaryWriter):
        pass

    @property
    def device(self) -> str:
        return "cuda" if next(self._policy_net.parameters()).is_cuda else "cpu"


def encode_card_set(card_set: List[Card]) -> torch.Tensor:
    return tensor([card in card_set for card in CARDS]).float()


def set_all_seeds(seed: int = 1988):
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)
