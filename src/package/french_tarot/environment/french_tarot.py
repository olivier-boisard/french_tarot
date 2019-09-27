import copy
from typing import List, Tuple, Union

import numpy as np
from tensorboard.backend.event_processing.event_file_inspector import Observation

from french_tarot.agents.meta import singledispatchmethod
from french_tarot.environment.core import Card, CARDS, count_trumps_and_excuse, rotate_list
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseEnvironment
from french_tarot.environment.subenvironments.card_phase import CardPhaseEnvironment
from french_tarot.environment.subenvironments.core import SubEnvironment
from french_tarot.environment.subenvironments.dog_phase import DogPhaseEnvironment
from french_tarot.exceptions import FrenchTarotException


class FrenchTarotEnvironment:
    n_players = 4
    dog_size = 6

    def __init__(self, seed: int = 1988):
        self._random_state = np.random.RandomState(seed)
        self._current_phase_environment: Union[SubEnvironment, None] = None
        self._hand_per_player = []
        self._made_dog = []
        self._bid_per_player = []
        self._announcements = []
        self._chelem_announced = False

    def reset(self):
        self._deal_until_valid()
        self._made_dog = []
        self._bid_per_player = []
        self._announcements = []
        self._chelem_announced = False
        self._initialize_current_phase_environment()
        observation = self._current_phase_environment.reset()
        return observation

    def step(self, action) -> Tuple[Observation, Union[float, List[float]], bool, any]:
        observation, reward, phase_is_done, info = self._current_phase_environment.step(action)

        done = False
        if phase_is_done:
            done = self._current_phase_environment.game_is_done
            observation = self._move_to_next_phase(self._current_phase_environment)

        if done:
            reward = rotate_list(reward, self.starting_player_id)
        return copy.deepcopy(observation), reward, done, info

    def render(self, mode="human", close=False):
        raise NotImplementedError()

    @property
    def starting_player_id(self):
        if self._chelem_announced:
            starting_player_id = 0
        else:
            taker_id = self._taker_original_id
            starting_player_id = list(np.arange(taker_id, taker_id + self.n_players) % self.n_players).index(0)
        return starting_player_id

    @property
    def original_player_ids(self):
        return (np.arange(self.n_players) + self._taker_original_id) % self.n_players

    @property
    def n_cards_per_player(self):
        return int((len(CARDS) - self.dog_size) // self.n_players)

    def _initialize_current_phase_environment(self):
        self._current_phase_environment = BidPhaseEnvironment(self._hand_per_player)
        self._current_phase_environment.reset()

    def _deal_until_valid(self):
        while True:
            try:
                deck = list(self._random_state.permutation(CARDS))
                self._deal(deck)
                break
            except RuntimeError as e:
                print(e)

    @singledispatchmethod
    def _move_to_next_phase(self, observation: SubEnvironment) -> Observation:
        raise FrenchTarotException("Unhandled type")

    @_move_to_next_phase.register
    def _(self, bid_phase_environment: BidPhaseEnvironment) -> Observation:
        self._taker_original_id = bid_phase_environment.taker_original_id
        self._shift_players_so_that_taker_has_id_0()
        self._bid_per_player = bid_phase_environment.bid_per_player
        if not bid_phase_environment.skip_dog_phase:
            observation = self._move_to_dog_phase()
        else:
            observation = self._move_to_announcement_phase()
        return observation

    @_move_to_next_phase.register
    def _(self, dog_phase_environment: DogPhaseEnvironment) -> Observation:
        self._made_dog = dog_phase_environment.new_dog
        self._hand_per_player[0] = dog_phase_environment.hand
        return self._move_to_announcement_phase()

    @_move_to_next_phase.register
    def _(self, announcement_phase_environment: AnnouncementPhaseEnvironment) -> Observation:
        self._announcements = announcement_phase_environment.announcements
        self._chelem_announced = announcement_phase_environment.chelem_announced
        return self._move_to_card_phase()

    @_move_to_next_phase.register
    def _(self, announcement_phase_environment: CardPhaseEnvironment) -> Observation:
        pass

    def _move_to_dog_phase(self) -> Observation:
        self._current_phase_environment = DogPhaseEnvironment(self._hand_per_player[0], self._original_dog)
        return self._current_phase_environment.reset()

    def _move_to_announcement_phase(self) -> Observation:
        self._current_phase_environment = AnnouncementPhaseEnvironment(self._hand_per_player)
        return self._current_phase_environment.reset()

    def _move_to_card_phase(self) -> Observation:
        self._current_phase_environment = CardPhaseEnvironment(
            self._hand_per_player,
            self.starting_player_id,
            self._made_dog,
            self._original_dog,
            self._bid_per_player,
            self._announcements
        )
        observation = self._current_phase_environment.reset()
        return observation

    def _shift_players_so_that_taker_has_id_0(self):
        self._hand_per_player = rotate_list(self._hand_per_player, -self._taker_original_id)

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
        for start in range(0, len(deck) - self.dog_size, self.n_cards_per_player):
            self._hand_per_player.append(deck[start:start + self.n_cards_per_player])
        self._check_player_hands()
