from typing import List, Tuple

import numpy as np
from attr import dataclass

from french_tarot.agents.meta import singledispatchmethod
from french_tarot.environment.core import Card, Announcement, PoigneeAnnouncement, count_trumps_and_excuse, \
    ChelemAnnouncement, Observation, PlayerData
from french_tarot.environment.subenvironments.core import SubEnvironment
from french_tarot.exceptions import FrenchTarotException


@dataclass
class AnnouncementPhaseObservation(Observation):
    pass


class AnnouncementPhaseEnvironment(SubEnvironment):

    def __init__(self, hand_per_player: List[List[Card]]):
        self._hand_per_player = hand_per_player
        self.announcements = []
        self.current_player = 0

    def reset(self):
        self.announcements = []
        return self.observation

    @property
    def game_is_done(self):
        return False

    @property
    def n_players(self):
        return len(self._hand_per_player)

    def step(self, action: List[Announcement]) -> Tuple[AnnouncementPhaseObservation, float, bool, any]:
        self._check(action)
        self.announcements.append(action)

        reward = 0
        info = None
        return self.observation, reward, self.done, info

    def _check(self, action):
        self._check_input_is_list(action)
        self._check_list_elements(action)
        self._check_player_announced_at_most_one_poignee(action)

    @staticmethod
    def _check_player_announced_at_most_one_poignee(action):
        if np.sum([isinstance(announcement, list) for announcement in action]) > 1:
            raise FrenchTarotException("Player tried to announcement more than 1 poignees")

    def _check_list_elements(self, action):
        for announcement in action:
            self._check_announcement(announcement)

    @staticmethod
    def _check_input_is_list(action):
        if not isinstance(action, list):
            raise FrenchTarotException("Input should be list")

    @singledispatchmethod
    def _check_announcement(self, announcement: Announcement):
        raise FrenchTarotException("Unhandled announcement")

    @_check_announcement.register
    def _(self, announcement: PoigneeAnnouncement):
        current_player_hand = self._hand_per_player[self.current_player_id]
        n_cards = len(announcement)
        if count_trumps_and_excuse(announcement.revealed_cards) != n_cards:
            raise FrenchTarotException("Invalid cards in poignee")
        n_trumps_in_hand = count_trumps_and_excuse(current_player_hand)
        if Card.EXCUSE in announcement.revealed_cards and n_trumps_in_hand != n_cards:
            raise FrenchTarotException(
                "Excuse can be revealed only if player does not have any other trumps")
        elif count_trumps_and_excuse(announcement.revealed_cards) != n_cards:
            raise FrenchTarotException("Revealed cards should be only trumps or excuse")
        elif np.any([card not in current_player_hand for card in announcement.revealed_cards]):
            raise FrenchTarotException("Revealed card not owned by player")

    @_check_announcement.register
    def _(self, _: ChelemAnnouncement):
        if len(self.announcements) > 0:
            raise FrenchTarotException("Only taker can announce chelem")

    @property
    def chelem_announced(self):
        is_chelem_announced = False
        for player_announcement in self.announcements:
            for announcement in player_announcement:
                if isinstance(announcement, ChelemAnnouncement):
                    is_chelem_announced = True
                    break
            if is_chelem_announced:
                break
        return is_chelem_announced

    @property
    def current_player_id(self):
        return len(self.announcements) % self.n_players

    @property
    def current_player_hand(self):
        return self._hand_per_player[self.current_player_id]

    @property
    def done(self):
        return len(self.announcements) == len(self._hand_per_player)

    @property
    def observation(self):
        current_player_data = PlayerData(self.current_player_id, self.current_player_hand)
        return AnnouncementPhaseObservation(current_player_data)

    def _get_next_player(self) -> int:
        return (self.current_player_id + 1) % self.n_players
