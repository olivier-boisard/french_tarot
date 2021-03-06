import numpy as np

from french_tarot.agents.agent import Agent, ActionWithProbability
from french_tarot.agents.card_phase_observation_encoder import retrieve_allowed_cards
from french_tarot.environment.core.announcements.chelem_announcement import ChelemAnnouncement
from french_tarot.environment.core.announcements.poignee.poignee_announcement import PoigneeAnnouncement
from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.core import is_oudler, get_minimum_allowed_bid, BIDS
from french_tarot.environment.subenvironments.announcements.announcements_phase_observation import \
    AnnouncementsPhaseObservation
from french_tarot.environment.subenvironments.bid.bid_phase_observation import BidPhaseObservation
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation
from french_tarot.environment.subenvironments.dog.dog_phase_observation import DogPhaseObservation
from french_tarot.exceptions import FrenchTarotException
from french_tarot.meta import singledispatchmethod


class RandomAgent(Agent):

    def __init__(self, seed: int = 1988):
        super().__init__()
        self._random_state = np.random.RandomState(seed)

    @singledispatchmethod
    def get_action(self, observation: any):
        raise FrenchTarotException("Unhandled game phase")

    @get_action.register
    def _(self, observation: CardPhaseObservation) -> ActionWithProbability:
        return self._random_state.choice(retrieve_allowed_cards(observation))

    @get_action.register
    def _(self, observation: AnnouncementsPhaseObservation) -> ActionWithProbability:
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
        return observation.player.position_towards_taker == 0

    @get_action.register
    def _(self, observation: DogPhaseObservation) -> ActionWithProbability:
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
    def _(self, observation: BidPhaseObservation) -> ActionWithProbability:
        allowed_bids = list(range(get_minimum_allowed_bid(observation.bid_per_player), np.max(BIDS) + 1))
        return Bid(self._random_state.choice(allowed_bids + [0]))
