import numpy as np

from agents.common import Agent
from environment import Bid, get_minimum_allowed_bid, GamePhase, CHELEM, TRIPLE_POIGNEE_SIZE, \
    get_trumps_and_excuse, Card, DOUBLE_POIGNEE_SIZE, SIMPLE_POIGNEE_SIZE, check_card_is_allowed, _is_oudler


def sort_trump_and_excuse(trumps_and_excuse):
    values = [int(card.value.split("_")[1]) if card != Card.EXCUSE else 22 for card in trumps_and_excuse]
    sorted_indexes = np.argsort(values)
    return list(trumps_and_excuse[sorted_indexes])


class RandomPlayer(Agent):

    def __init__(self, seed=1988):
        self._random_state = np.random.RandomState(seed)

    def optimize_model(self):
        pass  # overrides super class method to do nothing

    def get_action(self, observation):
        if observation["game_phase"] == GamePhase.BID:
            allowed_bids = list(range(get_minimum_allowed_bid(observation["bid_per_player"]), np.max(list(Bid)) + 1))
            rval = Bid(self._random_state.choice(allowed_bids + [0]))
        elif observation["game_phase"] == GamePhase.DOG:
            permuted_hand = self._random_state.permutation(observation["hand"])
            trump_allowed = False
            rval = []
            dog_size = len(observation["original_dog"])
            while len(rval) < dog_size:
                for card in permuted_hand:
                    if not _is_oudler(card) and "king" not in card.value:
                        if ("trump" in card.value and trump_allowed) or "trump" not in card.value:
                            rval.append(card)
                            if len(rval) == dog_size:
                                break
                trump_allowed = True
            assert len(rval) == dog_size
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
            card_list = np.array(Card)
            allowed_cards = []
            for card in card_list:
                try:
                    check_card_is_allowed(card, observation["played_cards"], observation["hand"])
                    allowed_cards.append(True)
                except ValueError:
                    allowed_cards.append(False)
            assert 1 <= np.sum(allowed_cards) <= len(observation["hand"])
            rval = self._random_state.choice(card_list[allowed_cards])
        else:
            raise ValueError("Unhandled game phase")
        return rval
