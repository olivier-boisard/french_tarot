import functools
from abc import ABC, abstractmethod
from typing import List

from french_tarot.environment.core.announcements.announcement import Announcement
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import sort_trump_and_excuse, get_trumps_and_excuse
from french_tarot.exceptions import FrenchTarotException


class PoigneeAnnouncement(Announcement, ABC):

    def __init__(self, revealed_cards: List[Card]):
        if len(revealed_cards) != self.expected_length():
            raise FrenchTarotException("Invalid number of cards")
        self.revealed_cards = revealed_cards

    def __len__(self):
        return len(self.revealed_cards)

    @staticmethod
    def largest_possible_poignee_factory(hand):
        trumps_and_excuse = sort_trump_and_excuse(get_trumps_and_excuse(hand))
        valid_poignees = filter(
            lambda element: element.expected_length() <= len(trumps_and_excuse),
            PoigneeAnnouncement.__subclasses__()
        )
        poignee = None
        cls = next(valid_poignees, None)
        if cls is not None:
            cls = functools.reduce(
                lambda a, b: a if a.expected_length() > b.expected_length() else b,
                valid_poignees,
                cls
            )

            revealed_cards_start_idx = 0
            # noinspection PyUnresolvedReferences
            revealed_cards_stop_idx = cls.expected_length()
            if len(trumps_and_excuse) > revealed_cards_stop_idx and Card.EXCUSE in trumps_and_excuse:
                trumps_and_excuse.remove(Card.EXCUSE)
            if Card.TRUMP_1 in trumps_and_excuse and revealed_cards_stop_idx != len(trumps_and_excuse):
                revealed_cards_start_idx += 1
                revealed_cards_stop_idx += 1
            poignee = cls(trumps_and_excuse[revealed_cards_start_idx:revealed_cards_stop_idx])

        return poignee

    @staticmethod
    @abstractmethod
    def expected_length():
        pass

    @staticmethod
    @abstractmethod
    def bonus_points():
        pass
