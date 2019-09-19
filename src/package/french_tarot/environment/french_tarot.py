import copy
from typing import List, Tuple, Union

import numpy as np

from french_tarot.environment.core import Card, GamePhase, Bid, CARDS, ChelemAnnouncement, \
    count_trumps_and_excuse
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation, \
    AnnouncementPhaseEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseEnvironment
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation, CardPhaseEnvironment
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation, DogPhaseEnvironment
from french_tarot.exceptions import FrenchTarotException


class FrenchTarotEnvironment:

    def __init__(self, seed: int = 1988):
        self._winners_per_round = None
        self._random_state = np.random.RandomState(seed)
        self._n_cards_in_dog = 6
        self._hand_per_player = []
        self._original_dog = []
        self._game_phase = None
        self._bid_per_player = None
        self.n_players = 4
        self._n_cards_per_player = int(np.round((len(CARDS) - self._n_cards_in_dog) / self.n_players))
        self._revealed_cards_in_dog = None
        self._announcements = None
        self.current_player = None
        self._played_cards_in_round = None
        self._won_cards_per_teams = None
        self._bonus_points_per_teams = None
        self._made_dog = None
        self.original_player_ids = None
        self._current_phase_environment = None
        self.taking_player_original_id = None
        self._starting_player = None
        self._past_rounds = []

    def reset(self):
        while True:
            try:
                deck = list(self._random_state.permutation(CARDS))
                self._deal(deck)
                break
            except RuntimeError as e:
                print(e)
        self._game_phase = GamePhase.BID
        self._bid_per_player = []
        self.n_players = 4
        self._announcements = []
        self.current_player = 0
        self._played_cards_in_round = []
        self._made_dog = None
        self._winners_per_round = []
        self.original_player_ids = []
        self._past_rounds = []

        self._current_phase_environment = BidPhaseEnvironment(self._hand_per_player, self._original_dog)

        return self._get_observation()

    # TODO create smaller functions
    def step(self, action) -> Tuple[any, Union[float, List[float]], bool, any]:
        # TODO create and use function overloading, or use dictionary
        done = False
        if self._game_phase == GamePhase.BID:
            reward, bid_phase_done, info = self._current_phase_environment.step(action)
            self.original_player_ids = self._current_phase_environment.original_player_ids
            self._bid_per_player = self._current_phase_environment.bid_per_player
            self._starting_player = self._current_phase_environment.next_phase_starting_player
            self.taking_player_original_id = self._current_phase_environment.taking_player_original_id
            self.current_player = self._current_phase_environment.current_player
            if bid_phase_done:
                done = self._current_phase_environment.all_players_passed
                if done:
                    reward = [0, 0, 0, 0]
                else:
                    self._game_phase = self._current_phase_environment.next_game_phase
                    if self._game_phase == GamePhase.ANNOUNCEMENTS:
                        self._current_phase_environment = AnnouncementPhaseEnvironment(
                            self._hand_per_player,
                            self._starting_player
                        )
            else:
                done = False
        elif self._game_phase == GamePhase.DOG:
            self._current_phase_environment = DogPhaseEnvironment(self._hand_per_player[0], self._original_dog)
            reward, done, info = self._current_phase_environment.step(action)
            self.current_player = self._starting_player
            self._hand_per_player[0] = self._current_phase_environment.hand
            self._made_dog = self._current_phase_environment.made_dog
            self._game_phase = GamePhase.ANNOUNCEMENTS
            self._current_phase_environment = AnnouncementPhaseEnvironment(self._hand_per_player, self._starting_player)
        elif self._game_phase == GamePhase.ANNOUNCEMENTS:
            reward, announcement_phase_done, info = self._current_phase_environment.step(action)
            self._announcements = self._current_phase_environment.announcements
            if announcement_phase_done:
                self._game_phase = GamePhase.CARD
                self.current_player = self._starting_player if not self.chelem_announced else 0
                self._current_phase_environment = CardPhaseEnvironment(
                    self._hand_per_player,
                    self._starting_player,
                    self._made_dog,
                    self._original_dog,
                    self._bid_per_player,
                    self._announcements,
                )
        elif self._game_phase == GamePhase.CARD:
            self._current_phase_environment.current_player = self.current_player
            self._current_phase_environment._hand_per_player = self._hand_per_player
            reward, done, info = self._current_phase_environment.step(action)
            self.current_player = self._current_phase_environment.current_player
            self._played_cards_in_round = self._current_phase_environment._played_cards_in_round
            self._past_rounds = self._current_phase_environment._past_rounds
            self._hand_per_player = self._current_phase_environment._hand_per_player
        else:
            raise RuntimeError("Unknown game phase")
        return self._get_observation(), reward, done, info

    @property
    def chelem_announced(self):
        return np.any([isinstance(announcement, ChelemAnnouncement) for announcement in self._announcements[0]])

    def _get_observation(self):
        # TODO fix duplications
        if self._game_phase == GamePhase.BID:
            observation = self._current_phase_environment.observation
        else:
            original_dog = self._original_dog if np.max(self._bid_per_player) <= Bid.GARDE else "unrevealed"
            # TODO use overloading
            if self._game_phase == GamePhase.DOG:
                observation = DogPhaseObservation(self._hand_per_player[0] + original_dog, len(original_dog))
            elif self._game_phase == GamePhase.ANNOUNCEMENTS:
                observation = AnnouncementPhaseObservation(len(self._announcements), self.current_hand)
            elif self._game_phase == GamePhase.CARD:
                observation = CardPhaseObservation(self.current_hand, self._played_cards_in_round)
            else:
                raise RuntimeError("Unknown game phase")
        return copy.deepcopy(observation)

    @property
    def current_hand(self):
        return self._hand_per_player[self.current_player]

    @current_hand.setter
    def current_hand(self, hand):
        self._hand_per_player[self.current_player] = hand


    def _get_next_player(self) -> int:
        return (self.current_player + 1) % self.n_players


    def _deal(self, deck: List[Card]):
        if len(deck) != len(CARDS):
            raise FrenchTarotException("Deck has wrong number of cards")
        del self._hand_per_player[:]
        self._hand_per_player.append(deck[:self._n_cards_per_player])
        self._hand_per_player.append(deck[self._n_cards_per_player:2 * self._n_cards_per_player])
        self._hand_per_player.append(deck[2 * self._n_cards_per_player:3 * self._n_cards_per_player])
        self._hand_per_player.append(deck[3 * self._n_cards_per_player:4 * self._n_cards_per_player])

        del self._original_dog[:]
        self._original_dog.extend(deck[-self._n_cards_in_dog:])
        for hand in self._hand_per_player:
            if Card.TRUMP_1 in hand and count_trumps_and_excuse(hand) == 1:
                raise RuntimeError("'Petit sec'. Deal again.")

    def render(self, mode="human", close=False):
        raise NotImplementedError()
