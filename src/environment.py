import copy
from collections import deque
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


CHELEM = "chelem"
SIMPLE_POIGNEE_SIZE = 10
DOUBLE_POIGNEE_SIZE = 13
TRIPLE_POIGNEE_SIZE = 15


def rotate_list(input_list, n):
    hand_per_player_deque = deque(input_list)
    hand_per_player_deque.rotate(n)
    return list(hand_per_player_deque)


class FrenchTarotEnvironment:
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self._random_state = np.random.RandomState(1988)
        self._n_cards_in_dog = 6
        self._hand_per_player = None
        self._original_dog = None
        self._game_phase = None
        self._bid_per_player = None
        self._n_players = None
        n_players = 4
        self._n_cards_per_player = int((len(list(Card)) - self._n_cards_in_dog) / n_players)
        self._revealed_cards_in_dog = None
        self._announcements = None
        self._chelem_announced = None
        self._current_player = None
        self._played_cards = None
        self._plis = None
        self._won_cards_per_teams = None
        self._bonus_points_per_teams = None
        self._made_dog = None

    def step(self, action):
        if self._game_phase == GamePhase.BID:
            reward, done, info = self._bid(action)
        elif self._game_phase == GamePhase.DOG:
            reward, done, info = self._make_dog(action)
        elif self._game_phase == GamePhase.ANNOUNCEMENTS:
            reward, done, info = self._announce(action)
        elif self._game_phase == GamePhase.CARD:
            reward, done, info = self._play_card(action)
        else:
            raise RuntimeError("Unknown game phase")
        return self._get_observation_for_current_player(), reward, done, info

    def _play_card(self, card):
        if not isinstance(card, Card):
            raise ValueError("Action must be card")
        if card not in self._hand_per_player[self._current_player]:
            raise ValueError("Card not in current player's hand")
        card_color = card.value.split("_")[0]
        asked_color = FrenchTarotEnvironment._retrieve_asked_color(self._played_cards)
        if card_color != asked_color and card != Card.EXCUSE and len(self._played_cards) > 0:
            self._check_trump_or_pee_is_allowed()
        if card_color == "trump":
            self._check_trump_value_is_allowed(card)

        self._played_cards.append(card)

        current_hand = self._hand_per_player[self._current_player]
        current_hand = current_hand[current_hand != card]
        rewards = None
        done = False
        if isinstance(current_hand, Card):
            self._hand_per_player[self._current_player] = np.array([current_hand])
        else:
            self._hand_per_player[self._current_player] = current_hand

        if len(self._played_cards) == self._n_players:
            rewards = self._solve_round()
            if len(self._hand_per_player[0]) == 0:
                rewards = self._compute_win_loss()
                done = True
            else:
                pass  # Nothing to do

        elif len(self._played_cards) < self._n_players:
            self._current_player = (self._current_player + 1) % self._n_players
        else:
            raise RuntimeError("Wrong number of played cards")

        info = None
        return rewards, done, info

    def _compute_win_loss(self):
        dog = self._made_dog if self._made_dog is not None else self._original_dog
        taker_points = get_card_set_point(self._won_cards_per_teams["taker"] + list(dog))
        taker_points += self._bonus_points_per_teams["taker"]
        opponents_points = get_card_set_point(self._won_cards_per_teams["opponents"])
        opponents_points += self._bonus_points_per_teams["opponents"]
        if taker_points + opponents_points != 91:
            raise RuntimeError("Invalid score")
        if taker_points != round(taker_points):
            raise RuntimeError("Score should be integer")
        n_oudlers_taker = np.sum([_is_oudler(card) for card in self._won_cards_per_teams["taker"]])
        if n_oudlers_taker == 3:
            victory_threshold = 36
        elif n_oudlers_taker == 2:
            victory_threshold = 41
        elif n_oudlers_taker == 1:
            victory_threshold = 51
        elif n_oudlers_taker == 0:
            victory_threshold = 56
        else:
            RuntimeError("Invalid number of oudlers")
        diff = abs(victory_threshold - taker_points)
        contract_value = 25 + diff
        if self._bid_per_player[0] == Bid.PETITE:
            multiplier = 1
        elif self._bid_per_player[0] == Bid.GARDE:
            multiplier = 2
        elif self._bid_per_player[0] == Bid.GARDE_SANS:
            multiplier = 4
        elif self._bid_per_player[0] == Bid.GARDE_CONTRE:
            multiplier = 6
        else:
            raise RuntimeError("Invalid contract value")
        contract_value = int(contract_value * multiplier)
        if taker_points < victory_threshold:
            contract_value *= -1
        else:
            pass  # Nothing to do
        rewards = [3 * contract_value, -contract_value, -contract_value, -contract_value]
        return rewards

    def _solve_round(self):
        starting_player = (self._current_player + 1) % self._n_players
        winning_card_index = FrenchTarotEnvironment._get_winning_card_index(self._played_cards)
        play_order = np.arange(starting_player, starting_player + self._n_players) % self._n_players
        winner = play_order[winning_card_index]
        reward_for_winner = get_card_set_point(self._played_cards)
        rewards = []
        if winner == 0:  # if winner is taking player
            rewards.append(reward_for_winner)
            rewards.extend([0] * (self._n_players - 1))
        else:
            rewards.append(0)
            rewards.extend([reward_for_winner] * (self._n_players - 1))
        self._plis.append({"played_cards": self._played_cards, "starting_player": starting_player})

        won_cards = self._played_cards.copy()
        if Card.EXCUSE in self._played_cards:
            excuse_owner = play_order[self._played_cards.index(Card.EXCUSE)]
            won_cards.remove(Card.EXCUSE)
            if excuse_owner == 0:
                self._won_cards_per_teams["taker"].append(Card.EXCUSE)
                self._bonus_points_per_teams["opponents"] += 0.5
                self._bonus_points_per_teams["taker"] -= 0.5
            else:
                self._won_cards_per_teams["opponents"].append(Card.EXCUSE)
                if winner == 0:
                    self._bonus_points_per_teams["opponents"] -= 0.5
                    self._bonus_points_per_teams["taker"] += 0.5
        if winner == 0:
            self._won_cards_per_teams["taker"] += won_cards
        else:
            self._won_cards_per_teams["opponents"] += won_cards
        self._current_player = winner
        self._played_cards = []
        return rewards

    def _check_trump_value_is_allowed(self, card):
        current_player_hand = self._hand_per_player[self._current_player]
        trumps_in_hand = [int(card.value.split("_")[1]) for card in current_player_hand if "trump" in card.value]
        max_trump_in_hand = np.max(trumps_in_hand) if len(trumps_in_hand) > 0 else 0
        played_trump = [int(card.value.split("_")[1]) for card in self._played_cards if "trump" in card.value]
        max_played_trump = np.max(played_trump) if len(played_trump) > 0 else 0
        card_strength = int(card.value.split("_")[1])
        if max_trump_in_hand > max_played_trump > card_strength:
            raise ValueError("Higher trump value must be played when possible")

    def _check_trump_or_pee_is_allowed(self):
        asked_color = FrenchTarotEnvironment._retrieve_asked_color(self._played_cards)
        if asked_color is not None:
            for card in self._hand_per_player[self._current_player]:
                if asked_color in card.value:
                    raise ValueError("Trump or pee unallowed")

    @staticmethod
    def _get_winning_card_index(played_cards):
        asked_color = FrenchTarotEnvironment._retrieve_asked_color(played_cards)
        card_strengths = []
        for card in played_cards:
            if "trump" in card.value:
                card_strengths.append(100 + int(card.value.split("_")[1]))
            elif asked_color not in card.value:
                card_strengths.append(0)
            elif "jack" in card.value:
                card_strengths.append(11)
            elif "rider" in card.value:
                card_strengths.append(12)
            elif "queen" in card.value:
                card_strengths.append(13)
            elif "king" in card.value:
                card_strengths.append(14)
            else:
                card_strengths.append(int(card.value.split("_")[1]))
        return np.argmax(card_strengths)

    @staticmethod
    def _retrieve_asked_color(played_cards):
        asked_color = None
        if len(played_cards) > 0:
            if played_cards[0] != Card.EXCUSE:
                asked_color = played_cards[0].value.split("_")[0]
            elif len(played_cards) > 1:
                asked_color = played_cards[1].value.split("_")[0]
            else:
                pass  # Nothing to do
        else:
            pass  # Nothing to do

        return asked_color

    def _announce(self, action: list):
        if not isinstance(action, list):
            raise ValueError("Input should be list")
        for announcement in action:
            if not isinstance(announcement, str) and not isinstance(announcement, list):
                raise ValueError("Wrong announcement type")
            elif isinstance(announcement, str) and announcement != CHELEM:
                raise ValueError("Wrong string value")
            elif isinstance(announcement, list):
                current_player_hand = self._hand_per_player[len(self._announcements)]
                n_trumps_in_hand = FrenchTarotEnvironment._count_trumps_and_excuse(current_player_hand)
                if Card.EXCUSE in announcement and n_trumps_in_hand != len(announcement):
                    raise ValueError("Excuse can be revealed only if player does not have any other trumps")
                elif FrenchTarotEnvironment._count_trumps_and_excuse(announcement) != len(announcement):
                    raise ValueError("Revealed cards should be only trumps or excuse")
                elif np.any([card not in current_player_hand for card in announcement]):
                    raise ValueError("Revealed card not owned by player")
            if announcement == CHELEM:
                if self._chelem_announced:
                    raise ValueError("Two players cannot announce chelems")
                else:
                    self._chelem_announced = True
                    self._current_player = len(self._announcements)

        if np.any([not isinstance(e, str) and not isinstance(e, list) for e in action]):
            raise ValueError("Wrong announcement type")
        if np.sum([isinstance(announcement, list) for announcement in action]) > 1:
            raise ValueError("Player tried to announcement more than 1 poignees")

        self._announcements.append(action)
        if len(self._announcements) == 4:
            self._game_phase = GamePhase.CARD
        else:
            pass  # Nothing to do

        reward = 0
        done = False
        info = None
        return reward, done, info

    def _make_dog(self, dog: list):
        taking_player_hand = self._hand_per_player[0]  # At this point, taking player is always player 0
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
            n_trumps_in_taking_player_hand = FrenchTarotEnvironment._count_trumps_and_excuse(taking_player_hand)
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
        self._hand_per_player[0] = taking_player_hand[index_to_keep_in_hand]

        reward = get_card_set_point(dog)
        self._made_dog = dog
        done = False
        info = None
        return reward, done, info

    @staticmethod
    def _count_trumps_and_excuse(cards):
        card_is_trump = np.array(["trump" in card.value or card.value == "excuse" for card in cards])
        n_trumps_in_taking_player_hand = np.sum(card_is_trump)
        return n_trumps_in_taking_player_hand

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
            taking_player = np.argmax(self._bid_per_player)
            self._hand_per_player = rotate_list(self._hand_per_player, -taking_player)
            self._current_player = -taking_player % self._n_players
            if np.max(self._bid_per_player) <= Bid.GARDE:
                self._hand_per_player[0] = np.concatenate((self._hand_per_player[0], self._original_dog))
                self._game_phase = GamePhase.DOG
            else:
                self._game_phase = GamePhase.ANNOUNCEMENTS
        else:
            done = False
        info = None
        return reward, done, info

    def reset(self):
        deck = self._random_state.permutation(list(Card))
        while True:
            try:
                self._deal(deck)
                break
            except RuntimeError as e:
                print(e)
        self._game_phase = GamePhase.BID
        self._bid_per_player = []
        self._n_players = 4
        self._announcements = []
        self._chelem_announced = False
        self._current_player = 0
        self._played_cards = []
        self._plis = []
        self._won_cards_per_teams = {"taker": [], "opponents": []}
        self._bonus_points_per_teams = {"taker": 0., "opponents": 0.}
        self._made_dog = None

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
        self._original_dog = deck[-self._n_cards_in_dog:]
        for hand in self._hand_per_player:
            if Card.TRUMP_1 in hand and FrenchTarotEnvironment._count_trumps_and_excuse(hand) == 1:
                raise RuntimeError("'Petit sec'. Deal again.")

    def _get_observation_for_current_player(self):
        rval = {
            "hand": self._hand_per_player[len(self._bid_per_player) - 1],
            "bid_per_player": self._bid_per_player,
            "game_phase": self._game_phase
        }
        if self._game_phase >= GamePhase.DOG:
            rval["original_dog"] = self._original_dog if np.max(self._bid_per_player) <= Bid.GARDE else "unrevealed"
        if self._game_phase >= GamePhase.ANNOUNCEMENTS:
            rval["revealed_cards_in_dog"] = self._revealed_cards_in_dog
            rval["announcements"] = self._announcements
        if self._game_phase >= GamePhase.CARD:
            rval["played_cards"] = self._played_cards
            rval["plis"] = self._plis

        return copy.deepcopy(rval)

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
