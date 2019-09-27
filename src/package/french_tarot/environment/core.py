import functools
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, IntEnum
from typing import List

import numpy as np
from attr import dataclass

from french_tarot.exceptions import FrenchTarotException


class Card(Enum):
    SPADES_1 = "spades_1"
    SPADES_2 = "spades_2"
    SPADES_3 = "spades_3"
    SPADES_4 = "spades_4"
    SPADES_5 = "spades_5"
    SPADES_6 = "spades_6"
    SPADES_7 = "spades_7"
    SPADES_8 = "spades_8"
    SPADES_9 = "spades_9"
    SPADES_10 = "spades_10"
    SPADES_JACK = "spades_jack"
    SPADES_RIDER = "spades_rider"
    SPADES_QUEEN = "spades_queen"
    SPADES_KING = "spades_king"
    CLOVER_1 = "clover_1"
    CLOVER_2 = "clover_2"
    CLOVER_3 = "clover_3"
    CLOVER_4 = "clover_4"
    CLOVER_5 = "clover_5"
    CLOVER_6 = "clover_6"
    CLOVER_7 = "clover_7"
    CLOVER_8 = "clover_8"
    CLOVER_9 = "clover_9"
    CLOVER_10 = "clover_10"
    CLOVER_JACK = "clover_jack"
    CLOVER_RIDER = "clover_rider"
    CLOVER_QUEEN = "clover_queen"
    CLOVER_KING = "clover_king"
    HEART_1 = "heart_1"
    HEART_2 = "heart_2"
    HEART_3 = "heart_3"
    HEART_4 = "heart_4"
    HEART_5 = "heart_5"
    HEART_6 = "heart_6"
    HEART_7 = "heart_7"
    HEART_8 = "heart_8"
    HEART_9 = "heart_9"
    HEART_10 = "heart_10"
    HEART_JACK = "heart_jack"
    HEART_RIDER = "heart_rider"
    HEART_QUEEN = "heart_queen"
    HEART_KING = "heart_king"
    DIAMOND_1 = "diamond_1"
    DIAMOND_2 = "diamond_2"
    DIAMOND_3 = "diamond_3"
    DIAMOND_4 = "diamond_4"
    DIAMOND_5 = "diamond_5"
    DIAMOND_6 = "diamond_6"
    DIAMOND_7 = "diamond_7"
    DIAMOND_8 = "diamond_8"
    DIAMOND_9 = "diamond_9"
    DIAMOND_10 = "diamond_10"
    DIAMOND_JACK = "diamond_jack"
    DIAMOND_RIDER = "diamond_rider"
    DIAMOND_QUEEN = "diamond_queen"
    DIAMOND_KING = "diamond_king"
    TRUMP_1 = "trump_1"
    TRUMP_2 = "trump_2"
    TRUMP_3 = "trump_3"
    TRUMP_4 = "trump_4"
    TRUMP_5 = "trump_5"
    TRUMP_6 = "trump_6"
    TRUMP_7 = "trump_7"
    TRUMP_8 = "trump_8"
    TRUMP_9 = "trump_9"
    TRUMP_10 = "trump_10"
    TRUMP_11 = "trump_11"
    TRUMP_12 = "trump_12"
    TRUMP_13 = "trump_13"
    TRUMP_14 = "trump_14"
    TRUMP_15 = "trump_15"
    TRUMP_16 = "trump_16"
    TRUMP_17 = "trump_17"
    TRUMP_18 = "trump_18"
    TRUMP_19 = "trump_19"
    TRUMP_20 = "trump_20"
    TRUMP_21 = "trump_21"
    EXCUSE = "excuse"


CARDS = list(Card)


class GamePhase(IntEnum):
    BID = 0
    DOG = 1
    ANNOUNCEMENTS = 2
    CARD = 3


class Bid(IntEnum):
    PASS = 0
    PETITE = 1
    GARDE = 2
    GARDE_SANS = 3
    GARDE_CONTRE = 4


class Announcement(ABC):
    pass


class ChelemAnnouncement(Announcement):
    pass


class Poignee:
    SIMPLE_POIGNEE_SIZE = 10
    DOUBLE_POIGNEE_SIZE = 13
    TRIPLE_POIGNEE_SIZE = 15


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
            poignee = cls(trumps_and_excuse[-cls.expected_length():])

        return poignee

    @staticmethod
    @abstractmethod
    def expected_length():
        pass

    @staticmethod
    @abstractmethod
    def bonus_points():
        pass


class SimplePoigneeAnnouncement(PoigneeAnnouncement):

    @staticmethod
    def expected_length():
        return 10

    @staticmethod
    def bonus_points():
        return 20


class DoublePoigneeAnnouncement(PoigneeAnnouncement):
    @staticmethod
    def expected_length():
        return 13

    @staticmethod
    def bonus_points():
        return 30


class TriplePoigneeAnnouncement(PoigneeAnnouncement):
    @staticmethod
    def expected_length():
        return 15

    @staticmethod
    def bonus_points():
        return 40


def sort_trump_and_excuse(trumps_and_excuse: List[Card]) -> List[Card]:
    values = [int(card.value.split("_")[1]) if card != Card.EXCUSE else 22 for card in trumps_and_excuse]
    sorted_indexes: np.array = np.argsort(values)
    return list(np.array(trumps_and_excuse)[sorted_indexes])


def get_trumps_and_excuse(cards: List[Card]) -> List[Card]:
    output_as_list = isinstance(cards, list)
    cards = np.array(cards)
    rval = cards[np.array(["trump" in card.value or card.value == "excuse" for card in cards])]
    if output_as_list:
        rval = list(rval)
    return rval


def rotate_list(input_list: List, n: int) -> List:
    hand_per_player_deque = deque(input_list)
    hand_per_player_deque.rotate(n)
    return list(hand_per_player_deque)


def get_minimum_allowed_bid(bid_per_player: List[Bid]) -> Bid:
    return Bid.PETITE if len(bid_per_player) == 0 else np.max(bid_per_player) + 1


def check_trump_or_pee_is_allowed(played_card: Card, played_cards_before: List[Card], player_hand: List[Card]):
    asked_color = retrieve_asked_color(played_cards_before)
    if asked_color is not None:
        for card in player_hand:
            if asked_color in card.value or ("trump" in card.value and "trump" not in played_card.value):
                raise FrenchTarotException("Trump or pee unallowed")


def check_trump_value_is_allowed(card: Card, played_cards_before: List[Card], current_player_hand: List[Card]):
    trumps_in_hand = [int(card.value.split("_")[1]) for card in current_player_hand if "trump" in card.value]
    max_trump_in_hand = np.max(trumps_in_hand) if len(trumps_in_hand) > 0 else 0
    played_trump = [int(card.value.split("_")[1]) for card in played_cards_before if "trump" in card.value]
    max_played_trump = np.max(played_trump) if len(played_trump) > 0 else 0
    card_strength = int(card.value.split("_")[1])
    if max_trump_in_hand > max_played_trump > card_strength:
        raise FrenchTarotException("Higher trump value must be played when possible")


def check_card_is_allowed(card: Card, played_cards: List[Card], player_hand: List[Card]):
    if card not in player_hand:
        raise FrenchTarotException("Card not in current player's hand")
    card_color = card.value.split("_")[0]
    asked_color = retrieve_asked_color(played_cards)
    if card_color != asked_color and card != Card.EXCUSE and len(played_cards) > 0:
        check_trump_or_pee_is_allowed(card, played_cards, player_hand)
    if card_color == "trump":
        check_trump_value_is_allowed(card, played_cards, player_hand)


def retrieve_asked_color(played_cards: List[Card]) -> str:
    asked_color = None
    if len(played_cards) > 0:
        if played_cards[0] != Card.EXCUSE:
            asked_color = played_cards[0].value.split("_")[0]
        elif len(played_cards) > 1:
            asked_color = played_cards[1].value.split("_")[0]

    return asked_color


def count_trumps_and_excuse(cards: List[Card]) -> int:
    trumps_and_excuse = get_trumps_and_excuse(cards)
    return len(trumps_and_excuse)


def is_oudler(card: Card) -> bool:
    return card == Card.TRUMP_1 or card == Card.TRUMP_21 or card == Card.EXCUSE


def get_card_point(card: Card) -> float:
    if is_oudler(card) or "king" in card.value:
        points = 4.5
    elif "queen" in card.value:
        points = 3.5
    elif "rider" in card.value:
        points = 2.5
    elif "jack" in card.value:
        points = 1.5
    else:
        points = 0.5
    return points


def get_card_set_point(card_list: List[Card]) -> float:
    # noinspection PyTypeChecker
    return float(np.sum([get_card_point(card) for card in card_list]))


@dataclass
class PlayerData:
    position_towards_taker: int
    hand: List[Card]


@dataclass
class Observation:
    player: PlayerData
