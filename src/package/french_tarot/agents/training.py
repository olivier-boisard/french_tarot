import math
import numpy as np


class Policy:
    def __init__(self, eps_start: float = 0.9, eps_end: float = 0.05, eps_decay: int = 500, random_seed: int = 1988):
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._random_state = np.random.RandomState(random_seed)

    def should_play_randomly(self, step):
        threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * step / self._eps_decay)
        plays_at_random = self._random_state.rand(1, 1) > threshold
        return plays_at_random
