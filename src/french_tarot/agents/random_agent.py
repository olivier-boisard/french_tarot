from typing import List

import numpy as np

from french_tarot.agents.common import Agent
from french_tarot.agents.meta import singledispatchmethod
from french_tarot.environment.common import Card, Bid, ChelemAnnouncement, PoigneeLength
from french_tarot.environment.environment import get_minimum_allowed_bid, get_trumps_and_excuse, \
    check_card_is_allowed, is_oudler
from french_tarot.environment.observations import Observation, BidPhaseObservation, DogPhaseObservation, \
    AnnouncementPhaseObservation, CardPhaseObservation


def sort_trump_and_excuse(trumps_and_excuse: List[Card]) -> List[Card]:
    values = [int(card.value.split("_")[1]) if card != Card.EXCUSE else 22 for card in trumps_and_excuse]
    sorted_indexes: np.array = np.argsort(values)
    return list(np.array(trumps_and_excuse)[sorted_indexes])


class RandomPlayer(Agent):

    def __init__(self, seed: int = 1988):
        super(RandomPlayer, self).__init__()
        self._random_state = np.random.RandomState(seed)

    @singledispatchmethod
    def get_action(self, observation: Observation):
        raise ValueError("Unhandled game phase")

    @get_action.register
    def _(self, observation: CardPhaseObservation):
        card_list = np.array(Card)
        allowed_cards = []
        for card in card_list:
            try:
                check_card_is_allowed(card, observation.played_cards_in_round, observation.hand)
                allowed_cards.append(True)
            except ValueError:
                allowed_cards.append(False)
        assert 1 <= np.sum(allowed_cards) <= len(observation.hand)
        rval = self._random_state.choice(card_list[allowed_cards])
        return rval

    @get_action.register
    def _(self, observation: AnnouncementPhaseObservation):
        announcements = []
        if len(observation.announcements) == 0 and self._random_state.rand(1, 1) < 0.1:
            announcements.append(ChelemAnnouncement())
        trumps_and_excuse = sort_trump_and_excuse(get_trumps_and_excuse(observation.hand))
        if len(trumps_and_excuse) >= PoigneeLength.TRIPLE_POIGNEE_SIZE:
            announcements.append(trumps_and_excuse[:PoigneeLength.TRIPLE_POIGNEE_SIZE])
        elif len(trumps_and_excuse) >= PoigneeLength.DOUBLE_POIGNEE_SIZE:
            announcements.append(trumps_and_excuse[:PoigneeLength.DOUBLE_POIGNEE_SIZE])
        elif len(trumps_and_excuse) >= PoigneeLength.SIMPLE_POIGNEE_SIZE:
            announcements.append(trumps_and_excuse[:PoigneeLength.SIMPLE_POIGNEE_SIZE])
        rval = announcements
        return rval

    @get_action.register
    def _(self, observation: DogPhaseObservation):
        permuted_hand = self._random_state.permutation(observation.hand)
        trump_allowed = False
        rval = []
        dog_size = len(observation.original_dog)
        while len(rval) < dog_size:
            for card in permuted_hand:
                if not is_oudler(card) and "king" not in card.value:
                    if ("trump" in card.value and trump_allowed) or "trump" not in card.value:
                        rval.append(card)
                        if len(rval) == dog_size:
                            break
            trump_allowed = True
        assert len(rval) == dog_size
        return rval

    @get_action.register
    def _(self, observation: BidPhaseObservation):
        allowed_bids = list(range(get_minimum_allowed_bid(observation.bid_per_player), np.max(list(Bid)) + 1))
        rval = Bid(self._random_state.choice(allowed_bids + [0]))
        return rval

    def optimize_model(self):
        pass
