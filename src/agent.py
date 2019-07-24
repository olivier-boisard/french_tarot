import numpy as np

from environment import Bid


class RandomPlayer:

    def __init__(self):
        self._random_state = np.random.RandomState(1988)

    def get_action(self, state):
        return Bid(self._random_state.choice(list(Bid)))
