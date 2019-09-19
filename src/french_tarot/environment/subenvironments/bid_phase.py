from typing import List, Tuple

import numpy as np
from attr import dataclass

from french_tarot.environment.core import Card, Bid, GamePhase, rotate_list, rotate_list_in_place, \
    get_minimum_allowed_bid
from french_tarot.environment.subenvironments.core import SubEnvironment
from french_tarot.exceptions import FrenchTarotException


@dataclass
class BidPhaseObservation:
    hand: List[Card]
    bid_per_player: List[Bid]


class BidPhaseEnvironment(SubEnvironment):

    def __init__(self, hand_per_player: List[List[Card]], original_dog: List[Card]):
        self._hand_per_player = hand_per_player
        self._original_dog = original_dog
        self.original_player_ids = None
        self.bid_per_player = []
        self.next_phase_starting_player = None
        self.taking_player_original_id = None
        self.current_player = 0

    @property
    def n_players(self):
        return len(self._hand_per_player)

    @property
    def all_players_passed(self):
        return np.all(np.array(self.bid_per_player) == Bid.PASS)

    # TODO create smaller functions
    def step(self, action: Bid) -> Tuple[float, bool, any]:
        self._check_input(action)
        self.bid_per_player.append(action)
        self.current_player = len(self.bid_per_player)

        if len(self.bid_per_player) == self.n_players:
            done = True
            # noinspection PyTypeChecker
            taking_player = int(np.argmax(self.bid_per_player))
            self._shift_taking_player_to_id_0(taking_player)
            self._check_state_consistency()
            if not self._dog_phase_should_be_skipped():
                self._prepare_for_dog_phase()
            else:
                self._prepare_for_announcement_phase()
        else:
            done = False

        info = None
        reward = 0
        return reward, done, info

    @property
    def observation(self) -> BidPhaseObservation:
        return BidPhaseObservation(self._hand_per_player[self.current_player], self.bid_per_player)

    def _dog_phase_should_be_skipped(self):
        skip_dog_phase = np.max(self.bid_per_player) > Bid.GARDE
        return skip_dog_phase

    def _prepare_for_announcement_phase(self):
        self.next_game_phase = GamePhase.ANNOUNCEMENTS
        self.current_player = 0  # taker makes announcements first

    def _prepare_for_dog_phase(self):
        self.next_game_phase = GamePhase.DOG
        self.current_player = self.next_phase_starting_player

    def _shift_taking_player_to_id_0(self, taking_player):
        original_player_ids = np.arange(taking_player, taking_player + self.n_players) % self.n_players
        self.original_player_ids = list(original_player_ids)
        self.bid_per_player = rotate_list(self.bid_per_player, -taking_player)
        rotate_list_in_place(self._hand_per_player, -taking_player)
        self.next_phase_starting_player = -taking_player % self.n_players
        self.taking_player_original_id = taking_player

    def _check_state_consistency(self):
        assert np.argmax(self.bid_per_player) == 0

    def _check_input(self, action):
        if type(action) != Bid:
            raise FrenchTarotException("Wrong type for 'action'")
        if action != Bid.PASS and action < get_minimum_allowed_bid(self.bid_per_player):
            raise FrenchTarotException("Action is not pass and is lower than highest bid")
