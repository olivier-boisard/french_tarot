import copy
from typing import List, Tuple, Union

import numpy as np

from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import Observation, rotate_list, CARDS, count_trumps_and_excuse
from french_tarot.environment.subenvironments.announcements.announcements_phase_environment import \
    AnnouncementPhaseEnvironment
from french_tarot.environment.subenvironments.bid.bid_phase_environment import BidPhaseEnvironment
from french_tarot.environment.subenvironments.card.card_phase_environment import CardPhaseEnvironment
from french_tarot.environment.subenvironments.dog.dog_phase_environment import DogPhaseEnvironment
from french_tarot.environment.subenvironments.sub_environment import SubEnvironment
from french_tarot.exceptions import FrenchTarotException
from french_tarot.meta import singledispatchmethod


class FrenchTarotEnvironment:
    n_players = 4
    _dog_size = 6

    def __init__(self, seed: int = 1988):
        self._random_state = np.random.RandomState(seed)
        self._current_phase_environment = None
        self._hand_per_player = None
        self._made_dog = None
        self._bid_per_player = None
        self._announcements = None
        self._chelem_announced = None

    def reset(self, shuffled_card_deck: List[Card] = None) -> Observation:
        if shuffled_card_deck is None:
            self._deal_until_valid()
        else:
            self._deal(shuffled_card_deck)
        self._made_dog = []
        self._initialize_first_phase_environment()
        observation = self._current_phase_environment.reset()
        return observation

    def step(self, action) -> Tuple[Observation, Union[float, List[float]], bool, any]:
        observation, reward, phase_is_done, info = self._current_phase_environment.step(action)

        done = False
        if phase_is_done:
            done = self._current_phase_environment.game_is_done
            observation = self._move_to_next_phase(self._current_phase_environment)

        if done:
            reward = rotate_list(reward, self._taker_id)
        return copy.deepcopy(observation), reward, done, info

    def render(self, mode="human", close=False):
        raise NotImplementedError()

    def extract_dog_phase_reward(self, rewards: List):
        max_bid = np.max(self._bid_per_player)
        reward = rewards[self._taker_id] if Bid.PASS < max_bid < Bid.GARDE_SANS else None
        return reward

    @property
    def _starting_player_position_towards_taker(self):
        """Returns the position of the starting player, counting from the taker, in the play order"""
        if self._chelem_announced:
            position = 0
        else:
            taker_id = self._taker_id
            position = list(np.arange(taker_id, taker_id + self.n_players) % self.n_players).index(0)
        return position

    @property
    def _n_cards_per_player(self):
        return int((len(CARDS) - self._dog_size) // self.n_players)

    def _initialize_first_phase_environment(self):
        self._current_phase_environment = BidPhaseEnvironment(self._hand_per_player)
        self._current_phase_environment.reset()

    def _deal_until_valid(self):
        while True:
            try:
                deck = list(self._random_state.permutation(CARDS))
                self._deal(deck)
                break
            except FrenchTarotException as e:
                print(e)

    @singledispatchmethod
    def _move_to_next_phase(self, observation: SubEnvironment) -> Observation:
        raise FrenchTarotException("Unhandled type")

    @_move_to_next_phase.register
    def _(self, bid_phase_environment: BidPhaseEnvironment) -> Observation:
        self._taker_id = bid_phase_environment.taker_id
        self._bid_per_player = bid_phase_environment.bid_per_player
        self._shift_players_so_that_taker_has_position_0()
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
            self._starting_player_position_towards_taker,
            self._made_dog,
            self._original_dog,
            self._bid_per_player,
            self._announcements
        )
        observation = self._current_phase_environment.reset()
        return observation

    def _shift_players_so_that_taker_has_position_0(self):
        self._hand_per_player = rotate_list(self._hand_per_player, -self._taker_id)

    def _deal(self, deck: List[Card]):
        if len(deck) != len(CARDS):
            raise FrenchTarotException("Deck has wrong number of cards")
        self._deal_to_players(deck)
        self._deal_to_dog(deck)

    def _deal_to_dog(self, deck):
        n_cards_in_dog = len(deck) - self.n_players * self._n_cards_per_player
        self._original_dog = deck[-n_cards_in_dog:]

    def _check_player_hands(self):
        for hand in self._hand_per_player:
            if Card.TRUMP_1 in hand and count_trumps_and_excuse(hand) == 1:
                raise FrenchTarotException("'Petit sec'. Deal again.")

    def _deal_to_players(self, deck):
        self._hand_per_player = []
        for start in range(0, len(deck) - self._dog_size, self._n_cards_per_player):
            self._hand_per_player.append(deck[start:start + self._n_cards_per_player])
        self._check_player_hands()
