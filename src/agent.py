import numpy as np

from environment import Bid, get_minimum_allowed_bid


class RandomPlayer:

    def __init__(self):
        self._random_state = np.random.RandomState(1988)

    def get_action(self, observation):
        allowed_bids = list(range(get_minimum_allowed_bid(observation["bid_per_player"]), np.max(list(Bid)) + 1))
        return Bid(self._random_state.choice(allowed_bids + [0]))
