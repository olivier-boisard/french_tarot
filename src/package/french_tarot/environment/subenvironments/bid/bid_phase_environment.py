from typing import List, Tuple

import numpy as np

from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import PlayerData, get_minimum_allowed_bid
from french_tarot.environment.subenvironments.bid.bid_phase_observation import BidPhaseObservation
from french_tarot.environment.subenvironments.sub_environment import SubEnvironment
from french_tarot.exceptions import FrenchTarotException


class BidPhaseEnvironment(SubEnvironment):

    def __init__(self, hand_per_player: List[List[Card]]):
        self._hand_per_player = hand_per_player
        self._initialize()

    def reset(self):
        self._initialize()
        return self.observation

    def step(self, action: Bid) -> Tuple[BidPhaseObservation, float, bool, any]:
        self._check(action)
        self.bid_per_player.append(action)

        info = None
        reward = [0] * self.n_players if self.game_is_done else 0
        return self.observation, reward, self.done, info

    def _initialize(self):
        self.bid_per_player = []

    @property
    def game_is_done(self):
        return self.done and self.all_players_passed

    @property
    def skip_dog_phase(self) -> bool:
        return np.any(np.array(self.bid_per_player) > Bid.GARDE)

    @property
    def done(self):
        return len(self.bid_per_player) == self.n_players

    @property
    def observation(self) -> BidPhaseObservation:
        current_player_data = PlayerData(self.current_player_id, self.current_player_hand)
        return BidPhaseObservation(current_player_data, self.bid_per_player)

    @property
    def current_player_id(self):
        return len(self.bid_per_player) % self.n_players

    @property
    def current_player_hand(self):
        return self._hand_per_player[self.current_player_id]

    @property
    def n_players(self):
        return len(self._hand_per_player)

    @property
    def taker_id(self):
        return np.argmax(np.array(self.bid_per_player))

    @property
    def all_players_passed(self):
        return np.all(np.array(self.bid_per_player) == Bid.PASS)

    def _check(self, action):
        if type(action) != Bid:
            raise FrenchTarotException("Wrong type for 'action'")
        if action != Bid.PASS and action < get_minimum_allowed_bid(self.bid_per_player):
            raise FrenchTarotException("Action is not pass and is lower than highest bid")
