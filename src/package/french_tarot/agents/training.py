import random
from collections import namedtuple
from typing import List

import math
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Policy:
    def __init__(self, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 500, random_seed: int = 0):
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._random_state = np.random.RandomState(random_seed)

    def should_play_randomly(self, step):
        threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * step / self._eps_decay)
        plays_at_random = self._random_state.rand(1, 1) > threshold
        return plays_at_random


class ReplayMemory:
    """
    Got from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity: int, random_seed: int = 0):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._random_state = random.Random(random_seed)

    def push_message(self, state: torch.Tensor, action: int, next_state: torch.Tensor, reward: float):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return self._random_state.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
