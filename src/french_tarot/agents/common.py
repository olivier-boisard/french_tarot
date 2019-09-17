import math
import random
from abc import abstractmethod, ABC
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from french_tarot.environment.common import Card, CARDS
from french_tarot.environment.observations import Observation

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Policy:
    def __init__(self, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 500, random_seed: int = 1988):
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._random_state = np.random.RandomState(random_seed)

    def should_play_randomly(self, step):
        threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * step / self._eps_decay)
        plays_at_random = self._random_state.rand() > threshold
        return plays_at_random


class ReplayMemory:
    """
    Got from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity: int, random_seed: int = 1988):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._random_state = random.Random(random_seed)

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


class CoreCardNeuralNet(nn.Module):
    N_CARDS_PER_COLOR = 14
    N_TRUMPS_AND_EXCUSES = 22

    def __init__(self):
        super().__init__()
        self._initialize_neural_net()

    def _initialize_neural_net(self, width=64):
        self.standard_cards_tower = nn.Sequential(
            nn.Linear(CoreCardNeuralNet.N_CARDS_PER_COLOR, width),
            nn.ReLU(),
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.ReLU()
        )
        self.trump_tower = nn.Sequential(
            nn.Linear(CoreCardNeuralNet.N_TRUMPS_AND_EXCUSES, width),
            nn.ReLU(),
            nn.BatchNorm1d(width),
            nn.Linear(width, width),
            nn.ReLU()
        )
        self.merge_tower = nn.Sequential(
            nn.BatchNorm1d(5 * width),
            nn.Linear(5 * width, 8 * width),
            nn.ReLU(),
            nn.BatchNorm1d(8 * width),
            nn.Linear(8 * width, 8 * width),
            nn.ReLU()
        )

    @property
    def output_dimensions(self) -> int:
        # noinspection PyUnresolvedReferences
        return self.merge_tower[-2].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = CoreCardNeuralNet.N_CARDS_PER_COLOR
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


class Trainer(ABC):

    def __init__(
            self,
            net: nn.Module,
            batch_size: int = 64,
            replay_memory_size: int = 20000,
            summary_writer: SummaryWriter = None,
            name=None
    ):
        self._net = net
        self._batch_size = batch_size
        self.memory = ReplayMemory(replay_memory_size)
        self._summary_writer = summary_writer
        self._name = name
        self._optimizer = Adam(net.parameters())
        self._initialize_inner_attribute()
        self._check_parameters()

    def _initialize_inner_attribute(self):
        self._step = 0
        self.loss = []

    def _check_parameters(self):
        if self._summary_writer is not None and self._name is None:
            raise ValueError("name should not be None if summary_writer is not None")

    @abstractmethod
    def push_to_memory(self, observation: Observation, action, reward):
        pass

    @abstractmethod
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_model_output_and_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def optimize_model(self):
        if len(self.memory) > self._batch_size:
            model_output, target = self.get_model_output_and_target()
            self._train_model_one_step(model_output, target)

    def _train_model_one_step(self, model_output: torch.Tensor, target: torch.Tensor):
        self._step += 1
        loss = self.compute_loss(model_output, target)
        self.loss.append(loss.item())

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), 0.1)
        self._optimizer.step()
        self._log()

    def _log(self):
        if self._step % 1000 == 0 and self._summary_writer is not None:
            self._summary_writer.add_scalar("Loss/train/" + self._name, self.loss[-1], self._step)

    @property
    def device(self) -> str:
        return "cuda" if next(self._net.parameters()).is_cuda else "cpu"


class BaseNeuralNetAgent(Agent, ABC):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net
        self._initialize_internals()

    def _initialize_internals(self):
        self._step = 0
        self._random_action_policy = Policy()

    def get_action(self, observation: Observation):
        self._step += 1
        if not self._random_action_policy.should_play_randomly(self._step):
            self.policy_net.eval()
            with torch.no_grad():
                action = self.get_max_return_action(observation)
            self.policy_net.train()
        else:
            action = self.get_random_action(observation)
        return action

    @abstractmethod
    def get_max_return_action(self, observation: Observation):
        pass

    @abstractmethod
    def get_random_action(self, observation: Observation):
        pass

    @property
    def device(self) -> str:
        return "cuda" if next(self.policy_net.parameters()).is_cuda else "cpu"


def encode_cards(cards: List[Card]) -> torch.Tensor:
    return tensor([card in cards for card in CARDS]).float()


def set_all_seeds(seed: int = 1988):
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)
