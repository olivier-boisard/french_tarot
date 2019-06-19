from enum import Enum

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


class FrenchTarotEnvironment:
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self._random_state = np.random.RandomState(1988)
        self._state = None

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        shuffled_deck = self._random_state.permutation(list(Card))
        n_cards_in_dog = 6
        n_players = 4
        n_cards_per_player = int((len(shuffled_deck) - n_cards_in_dog) / n_players)

        self._state = {
            "hand_per_player": [
                shuffled_deck[:n_cards_per_player],
                shuffled_deck[n_cards_per_player:2 * n_cards_per_player],
                shuffled_deck[2 * n_cards_per_player:3 * n_cards_per_player],
                shuffled_deck[3 * n_cards_per_player:4 * n_cards_per_player],
            ],
            "card_played_in_turn": [],
            "card_played_per_player_in_episode": [],
            "who_took": None,
            "is_dog_revealed": False,
            "dog": shuffled_deck[-n_cards_in_dog:],
            "is_game_started": False,
            "bid_per_player": [None, None, None, None],
            "announcements": []
        }
        return self._state

    def render(self, mode="human", close=False):
        raise NotImplementedError()
