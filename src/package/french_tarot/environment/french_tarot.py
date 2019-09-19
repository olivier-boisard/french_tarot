from typing import List, Tuple, Union

import numpy as np

from french_tarot.environment.core import Card, CARDS, count_trumps_and_excuse
from french_tarot.environment.subenvironments.bid_phase import BidPhaseEnvironment
from french_tarot.environment.subenvironments.core import SubEnvironment
from french_tarot.exceptions import FrenchTarotException


class FrenchTarotEnvironment:
    n_players = 4

    def __init__(self, seed: int = 1988):
        self._random_state = np.random.RandomState(seed)
        self.done = False
        self._current_phase_environment: Union[SubEnvironment, None] = None

    def reset(self):
        self._deal_until_valid()
        self._initialize_current_phase_environment()
        observation = self._current_phase_environment.reset()
        return observation

    def _initialize_current_phase_environment(self):
        self._current_phase_environment = BidPhaseEnvironment(self._hand_per_player)

    def _deal_until_valid(self):
        while True:
            try:
                deck = list(self._random_state.permutation(CARDS))
                self._deal(deck)
                break
            except RuntimeError as e:
                print(e)

    def step(self, action) -> Tuple[any, Union[float, List[float]], bool, any]:
        observation, reward, phase_is_done, info = self._current_phase_environment.step(action)

        if phase_is_done:
            self.done = self._current_phase_environment.game_is_done
            self._initialize_next_phase_environment()
        return observation, reward, self.done, info

    def _initialize_next_phase_environment(self):
        raise NotImplementedError()

    def _deal(self, deck: List[Card]):
        if len(deck) != len(CARDS):
            raise FrenchTarotException("Deck has wrong number of cards")
        self._deal_to_players(deck)
        self._deal_to_dog(deck)

    def _deal_to_dog(self, deck):
        n_cards_in_dog = len(deck) - self.n_players * self.n_cards_per_player
        self._original_dog = deck[-n_cards_in_dog:]

    def _check_player_hands(self):
        for hand in self._hand_per_player:
            if Card.TRUMP_1 in hand and count_trumps_and_excuse(hand) == 1:
                raise RuntimeError("'Petit sec'. Deal again.")

    def _deal_to_players(self, deck):
        self._hand_per_player = []
        for start in range(0, len(deck), self.n_cards_per_player):
            self._hand_per_player.append(deck[start:start + self.n_cards_per_player])
        self._check_player_hands()

    def render(self, mode="human", close=False):
        raise NotImplementedError()

    @property
    def n_cards_per_player(self):
        return int(len(CARDS) // self.n_players)
