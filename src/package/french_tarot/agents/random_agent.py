import numpy as np

from french_tarot.agents.agent import Agent
from french_tarot.environment.core import Bid, ChelemAnnouncement, PoigneeAnnouncement, get_minimum_allowed_bid, \
    check_card_is_allowed, is_oudler, CARDS, BIDS
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation
from french_tarot.exceptions import FrenchTarotException
from french_tarot.meta import singledispatchmethod


class RandomPlayer(Agent):

    def __init__(self, seed: int = 1988):
        super().__init__()
        self._random_state = np.random.RandomState(seed)

    @singledispatchmethod
    def get_action(self, observation: any):
        raise FrenchTarotException("Unhandled game phase")

    @get_action.register
    def _(self, observation: CardPhaseObservation):
        card_list = np.array(CARDS)
        allowed_cards = []
        for card in card_list:
            try:
                check_card_is_allowed(card, observation.played_cards_in_round, observation.player.hand)
                allowed_cards.append(True)
            except FrenchTarotException:
                allowed_cards.append(False)
        assert 1 <= np.sum(allowed_cards) <= len(observation.player.hand)
        action = self._random_state.choice(card_list[allowed_cards])
        return action

    @get_action.register
    def _(self, observation: AnnouncementPhaseObservation):
        announcements = []
        if self._announce_chelem(observation):
            announcements.append(ChelemAnnouncement())
        hand = observation.player.hand
        poignee = PoigneeAnnouncement.largest_possible_poignee_factory(hand)
        if poignee is not None:
            announcements.append(poignee)
        return announcements

    def _announce_chelem(self, observation):
        if self._player_is_taker(observation):
            chelem_announcement_probability = 0.1
            announce_chelem = self._random_state.rand(1, 1) < chelem_announcement_probability
        else:
            announce_chelem = False
        return announce_chelem

    @staticmethod
    def _player_is_taker(observation):
        player_is_taker = observation.player.position_towards_taker == 0
        return player_is_taker

    @get_action.register
    def _(self, observation: DogPhaseObservation):
        permuted_hand = self._random_state.permutation(observation.player.hand)
        trump_allowed = False
        action = []
        while len(action) < observation.dog_size:
            for card in permuted_hand:
                if not is_oudler(card) and "king" not in card.value:
                    if ("trump" in card.value and trump_allowed) or "trump" not in card.value:
                        action.append(card)
                        if len(action) == observation.dog_size:
                            break
            trump_allowed = True
        assert len(action) == observation.dog_size
        return action

    @get_action.register
    def _(self, observation: BidPhaseObservation):
        allowed_bids = list(range(get_minimum_allowed_bid(observation.bid_per_player), np.max(BIDS) + 1))
        action = Bid(self._random_state.choice(allowed_bids + [0]))
        return action
