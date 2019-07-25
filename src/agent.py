import numpy as np

from environment import Bid, get_minimum_allowed_bid, GamePhase, CHELEM, TRIPLE_POIGNEE_SIZE, \
    get_trumps_and_excuse, Card, DOUBLE_POIGNEE_SIZE, SIMPLE_POIGNEE_SIZE


def sort_trump_and_excuse(trumps_and_excuse):
    values = [int(card.value.split("_")[1]) if card != Card.EXCUSE else 0 for card in trumps_and_excuse]
    sorted_indexes = np.argsort(values)
    return trumps_and_excuse[sorted_indexes]


class RandomPlayer:

    def __init__(self):
        self._random_state = np.random.RandomState(1988)

    def get_action(self, observation):
        if observation["game_phase"] == GamePhase.BID:
            allowed_bids = list(range(get_minimum_allowed_bid(observation["bid_per_player"]), np.max(list(Bid)) + 1))
            rval = Bid(self._random_state.choice(allowed_bids + [0]))
        elif observation["game_phase"] == GamePhase.DOG:
            raise ValueError()
        elif observation["game_phase"] == GamePhase.ANNOUNCEMENTS:
            announcements = []
            if len(observation["announcements"]) == 0 and self._random_state.rand() < 0.1:
                announcements.append(CHELEM)
            else:
                pass  # Nothing to do

            trumps_and_excuse = sort_trump_and_excuse(get_trumps_and_excuse(observation["hand"]))
            if len(trumps_and_excuse) >= TRIPLE_POIGNEE_SIZE:
                announcements.append(trumps_and_excuse[:TRIPLE_POIGNEE_SIZE])
            elif len(trumps_and_excuse) >= DOUBLE_POIGNEE_SIZE:
                announcements.append(trumps_and_excuse[:DOUBLE_POIGNEE_SIZE])
            elif len(trumps_and_excuse) >= SIMPLE_POIGNEE_SIZE:
                announcements.append(trumps_and_excuse[:SIMPLE_POIGNEE_SIZE])
            else:
                pass  # Nothing to do

            rval = announcements
        elif observation["game_phase"] == GamePhase.CARD:

        else:
            raise ValueError("Unhandled game phase")
        return rval
