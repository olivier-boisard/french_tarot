from collections import deque
from typing import List

import numpy as np
from attr import dataclass

from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.card import Card
from french_tarot.exceptions import FrenchTarotException

# TODO move to Card enum
# noinspection PyTypeChecker
CARDS = list(Card)

# TODO move to Bid enum
# noinspection PyTypeChecker
BIDS = list(Bid)


def sort_trump_and_excuse(trumps_and_excuse: List[Card]) -> List[Card]:
    values = [int(card.value.split("_")[1]) if card != Card.EXCUSE else 22 for card in trumps_and_excuse]
    sorted_indexes: np.array = np.argsort(values)
    return list(np.array(trumps_and_excuse)[sorted_indexes])


def get_trumps_and_excuse(cards: List[Card]) -> List[Card]:
    output_as_list = isinstance(cards, list)
    cards = np.array(cards)
    trumps_and_excuses = cards[np.array(["trump" in card.value or card.value == "excuse" for card in cards])]
    if output_as_list:
        trumps_and_excuses = list(trumps_and_excuses)
    return trumps_and_excuses


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
                raise FrenchTarotException("Trump or pee not allowed")


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


def compute_card_set_points(card_list: List[Card]) -> float:
    # noinspection PyTypeChecker
    return float(np.sum([get_card_point(card) for card in card_list]))


@dataclass
class PlayerData:
    position_towards_taker: int
    hand: List[Card]


@dataclass
class Observation:
    player: PlayerData
