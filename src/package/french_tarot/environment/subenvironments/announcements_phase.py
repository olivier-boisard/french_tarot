from typing import List

import numpy as np
from attr import dataclass

from french_tarot.environment.core import Card, Announcement, PoigneeAnnouncement, count_trumps_and_excuse, \
    ChelemAnnouncement
from french_tarot.exceptions import FrenchTarotException


@dataclass
class AnnouncementPhaseObservation:
    current_player_id: int
    hand: List[Card]


class AnnouncementPhaseEnvironment:

    def __init__(self, hand_per_player: List[List[Card]], starting_player: int):
        self._hand_per_player = hand_per_player
        self.chelem_announced = False
        self.announcements = []
        self._starting_player = starting_player
        self.current_player = 0

    @property
    def n_players(self):
        return len(self._hand_per_player)

    def step(self, action: List[Announcement]):
        if not isinstance(action, list):
            raise FrenchTarotException("Input should be list")
        for announcement in action:
            # TODO use function overloading
            if not isinstance(announcement, Announcement):
                raise FrenchTarotException("Invalid action type")
            elif isinstance(announcement, PoigneeAnnouncement):
                current_player_hand = self._hand_per_player[self.current_player]
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
            elif isinstance(announcement, ChelemAnnouncement):
                if len(self.announcements) > 0:
                    raise FrenchTarotException("Only taker can announce chelem")

        if np.sum([isinstance(announcement, list) for announcement in action]) > 1:
            raise FrenchTarotException("Player tried to announcement more than 1 poignees")

        self.announcements.append(action)
        done = len(self.announcements) == self.n_players
        if done:
            self.current_player = 0 if self.chelem_announced else self._starting_player
        else:
            self.current_player = self._get_next_player()

        reward = 0
        info = None
        return reward, done, info

    def _get_next_player(self) -> int:
        return (self.current_player + 1) % self.n_players
