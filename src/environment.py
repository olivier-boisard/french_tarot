from enum import Enum, IntEnum

import numpy as np


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


class GamePhase(Enum):
    BID = "bid"
    DOG = "dog"
    ANNOUNCEMENTS = "announcements"
    CARD = "card"


class Bid(IntEnum):
    PASS = 0
    PETITE = 1
    GARDE = 2
    GARDE_SANS = 3
    GARDE_CONTRE = 4


class FrenchTarotEnvironment:
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self._random_state = np.random.RandomState(1988)
        self._n_cards_in_dog = 6
        self._hand_per_player = None
        self._dog = None
        self._game_phase = None
        self._bid_per_player = None
        self._n_players = None
        self._taking_player = None
        n_players = 4
        self._n_cards_per_player = int((len(list(Card)) - self._n_cards_in_dog) / n_players)
        self._revealed_cards_in_dog = None
        self._won_cards_per_team = None

    def step(self, action):
        if self._game_phase == GamePhase.BID:
            reward, done, info = self._bid(action)
        elif self._game_phase == GamePhase.DOG:
            reward, done, info = self._make_dog(action)
        else:
            RuntimeError("Unknown game phase")
        return self._get_observation_for_current_player(), reward, done, info

    def _make_dog(self, dog: list):
        taking_player_hand = self._hand_per_player[self._taking_player]
        if type(dog) != list:
            raise ValueError("Wrong type for 'action'")
        if len(set(dog)) != len(dog):
            raise ValueError("Duplicated cards in dog")
        if np.any(["king" in card.value for card in dog]):
            raise ValueError("There should be no king in dog")
        if np.any([_is_oudler(card) for card in dog]):
            raise ValueError("There should be no oudler in dog")
        if np.any([card not in taking_player_hand for card in dog]):
            raise ValueError("Card in dog not in taking player's hand")
        if len(dog) != self._n_cards_in_dog:
            raise ValueError("Wrong number of cards in dog")

        print(dog)
        n_trumps_in_dog = np.sum(["trump" in card.value for card in dog])
        if n_trumps_in_dog > 0:
            card_is_trump = np.array(["trump" in card.value or card.value == "excuse" for card in taking_player_hand])
            n_trumps_in_taking_player_hand = np.sum(card_is_trump)
            n_kings_in_taking_player_hand = np.sum(["king" in card.value for card in taking_player_hand])
            allowed_trumps_in_dog = self._n_cards_in_dog - (
                    len(taking_player_hand) - n_trumps_in_taking_player_hand - n_kings_in_taking_player_hand)
            if n_trumps_in_dog != allowed_trumps_in_dog:
                raise ValueError("There should be no more trumps in dog than needed")

            card_in_dog_is_trump = np.array(["trump" in card.value for card in dog])
            self._revealed_cards_in_dog = list(np.array(dog)[card_in_dog_is_trump])
        else:
            self._revealed_cards_in_dog = []

        self._game_phase = GamePhase.ANNOUNCEMENTS
        index_to_keep_in_hand = [card not in dog for card in taking_player_hand]
        self._hand_per_player[self._taking_player] = taking_player_hand[index_to_keep_in_hand]
        self._won_cards_per_team["taker"].extend(dog)

        reward = get_card_set_point(dog)
        done = False
        info = None
        return reward, done, info

    def _bid(self, action: Bid):
        if type(action) != Bid:
            raise ValueError("Wrong type for 'action'")

        if len(self._bid_per_player) > 0:
            if action != Bid.PASS and action <= np.max(self._bid_per_player):
                raise ValueError("Action is not pass and is lower than highest bid")
        self._bid_per_player.append(action)
        reward = 0
        if len(self._bid_per_player) == self._n_players:
            done = np.all(np.array(self._bid_per_player) == Bid.PASS)
            self._taking_player = np.argmax(self._bid_per_player)
            if np.max(self._bid_per_player) <= Bid.GARDE:
                self._hand_per_player[self._taking_player] = np.concatenate((self._hand_per_player[self._taking_player],
                                                                             self._dog))
                self._game_phase = GamePhase.DOG
            else:
                self._game_phase = GamePhase.ANNOUNCEMENTS
        else:
            done = False
        info = None
        return reward, done, info

    def reset(self):
        deck = self._random_state.permutation(list(Card))
        self._deal(deck)
        self._game_phase = GamePhase.BID
        self._bid_per_player = []
        self._n_players = 4
        self._won_cards_per_team = {"taker": [], "opponents": []}

        return self._get_observation_for_current_player()

    def _deal(self, deck):
        if len(deck) != len(list(Card)):
            raise ValueError("Deck has wrong number of cards")
        self._hand_per_player = [
            deck[:self._n_cards_per_player],
            deck[self._n_cards_per_player:2 * self._n_cards_per_player],
            deck[2 * self._n_cards_per_player:3 * self._n_cards_per_player],
            deck[3 * self._n_cards_per_player:4 * self._n_cards_per_player],
        ]
        self._dog = deck[-self._n_cards_in_dog:]

    def _get_observation_for_current_player(self):
        rval = {
            "hand": self._hand_per_player[len(self._bid_per_player) - 1],
            "bid_per_player": self._bid_per_player,
            "revealed_cards_in_dog": self._revealed_cards_in_dog,
            "game_phase": self._game_phase
        }
        return rval

    def render(self, mode="human", close=False):
        raise NotImplementedError()


def _is_oudler(card):
    return card == Card.TRUMP_1 or card == Card.TRUMP_21 or card == Card.EXCUSE


def get_card_point(card: Card):
    if _is_oudler(card) or "king" in card.value:
        rval = 4.5
    elif "queen" in card.value:
        rval = 3.5
    elif "rider" in card.value:
        rval = 2.5
    elif "jack" in card.value:
        rval = 1.5
    else:
        rval = 0.5
    return rval


def get_card_set_point(card_list: list):
    return np.sum([get_card_point(card) for card in card_list])
