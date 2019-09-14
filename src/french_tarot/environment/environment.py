import copy
from collections import deque
from typing import List, Tuple, Union

import numpy as np

from french_tarot.environment.common import Card, GamePhase, Bid, CARDS, PoigneeAnnouncement, Announcement, \
    ChelemAnnouncement
from french_tarot.environment.observations import BidPhaseObservation, DogPhaseObservation, \
    AnnouncementPhaseObservation, CardPhaseObservation, Round, Observation


def rotate_list(input_list: List, n: int) -> List:
    hand_per_player_deque = deque(input_list)
    hand_per_player_deque.rotate(n)
    return list(hand_per_player_deque)


def get_minimum_allowed_bid(bid_per_player: List[Bid]) -> Bid:
    return Bid.PETITE if len(bid_per_player) == 0 else np.max(bid_per_player) + 1


def check_trump_or_pee_is_allowed(played_card: Card, played_cards_before: List[Card], player_hand: List[Card]):
    asked_color = _retrieve_asked_color(played_cards_before)
    if asked_color is not None:
        for card in player_hand:
            if asked_color in card.value or ("trump" in card.value and "trump" not in played_card.value):
                raise ValueError("Trump or pee unallowed")


def check_trump_value_is_allowed(card: Card, played_cards_before: List[Card], current_player_hand: List[Card]):
    trumps_in_hand = [int(card.value.split("_")[1]) for card in current_player_hand if "trump" in card.value]
    max_trump_in_hand = np.max(trumps_in_hand) if len(trumps_in_hand) > 0 else 0
    played_trump = [int(card.value.split("_")[1]) for card in played_cards_before if "trump" in card.value]
    max_played_trump = np.max(played_trump) if len(played_trump) > 0 else 0
    card_strength = int(card.value.split("_")[1])
    if max_trump_in_hand > max_played_trump > card_strength:
        raise ValueError("Higher trump value must be played when possible")


def check_card_is_allowed(card: Card, played_cards: List[Card], player_hand: List[Card]):
    if card not in player_hand:
        raise ValueError("Card not in current player's hand")
    card_color = card.value.split("_")[0]
    asked_color = _retrieve_asked_color(played_cards)
    if card_color != asked_color and card != Card.EXCUSE and len(played_cards) > 0:
        check_trump_or_pee_is_allowed(card, played_cards, player_hand)
    if card_color == "trump":
        check_trump_value_is_allowed(card, played_cards, player_hand)


class FrenchTarotEnvironment:
    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int = 1988):
        self._winners_per_round = None
        self._random_state = np.random.RandomState(seed)
        self._n_cards_in_dog = 6
        self._hand_per_player = None
        self._original_dog = None
        self._game_phase = None
        self._bid_per_player = None
        self.n_players = 4
        self._n_cards_per_player = int((len(CARDS) - self._n_cards_in_dog) / self.n_players)
        self._revealed_cards_in_dog = None
        self._announcements = None
        self._chelem_announced = None
        self.current_player = None
        self._played_cards_in_round = None
        self._past_rounds = None
        self._won_cards_per_teams = None
        self._bonus_points_per_teams = None
        self._made_dog = None
        self._original_player_ids = None

    def step(self, action) -> Tuple[Observation, Union[float, List[float]], bool, any]:
        if self._game_phase == GamePhase.BID:
            reward, done, info = self._bid(action)
        elif self._game_phase == GamePhase.DOG:
            reward, done, info = self._make_dog(action)
            self.current_player = self._starting_player
        elif self._game_phase == GamePhase.ANNOUNCEMENTS:
            reward, done, info = self._announce(action)
        elif self._game_phase == GamePhase.CARD:
            reward, done, info = self._play_card(action)
        else:
            raise RuntimeError("Unknown game phase")
        return self._get_observation(), reward, done, info

    def _play_card(self, card: Card) -> Tuple[List[float], bool, any]:
        if not isinstance(card, Card):
            raise ValueError("Action must be card")
        check_card_is_allowed(card, self._played_cards_in_round, self._hand_per_player[self.current_player])
        self._played_cards_in_round.append(card)

        current_hand = self._hand_per_player[self.current_player]
        current_hand = list(np.array(current_hand)[np.array(current_hand) != card])
        rewards = None
        done = False
        if isinstance(current_hand, Card):
            self._hand_per_player[self.current_player] = list([current_hand])
        else:
            self._hand_per_player[self.current_player] = current_hand

        if len(self._played_cards_in_round) == self.n_players:
            is_petit_played_in_round = Card.TRUMP_1 in self._played_cards_in_round
            is_excuse_played_in_round = Card.EXCUSE in self._played_cards_in_round
            rewards = self._solve_round()
            is_taker_win_round = rewards[0] > 0
            if len(self._hand_per_player[0]) == 0:
                rewards = self._compute_win_loss(is_petit_played_in_round, is_excuse_played_in_round,
                                                 is_taker_win_round)
                done = True

        elif len(self._played_cards_in_round) < self.n_players:
            self.current_player = self._get_next_player()
        else:
            raise RuntimeError("Wrong number of played cards")

        info = None
        return rewards, done, info

    def _get_next_player(self) -> int:
        return (self.current_player + 1) % self.n_players

    def _compute_win_loss(self, is_petit_played_in_round: bool, is_excuse_played_in_round: bool,
                          is_taker_win_round: bool) -> List[float]:
        dog = self._made_dog if self._made_dog is not None else self._original_dog
        taker_points = get_card_set_point(self._won_cards_per_teams["taker"] + list(dog))
        taker_points += self._bonus_points_per_teams["taker"]
        opponents_points = get_card_set_point(self._won_cards_per_teams["opponents"])
        opponents_points += self._bonus_points_per_teams["opponents"]
        if taker_points + opponents_points != 91:
            raise RuntimeError("Invalid score")
        if taker_points != round(taker_points):
            raise RuntimeError("Score should be integer")
        n_oudlers_taker = np.sum([is_oudler(card) for card in list(self._won_cards_per_teams["taker"]) + list(dog)])
        victory_threshold = None
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

        winners_per_round = self._winners_per_round
        taker_achieved_chelem = self.has_team_achieved_chelem(winners_per_round, is_excuse_played_in_round, "taker")
        if taker_achieved_chelem:
            contract_value += 400 if self._chelem_announced else 200
        elif self.has_team_achieved_chelem(winners_per_round, is_excuse_played_in_round, "opponents"):
            contract_value -= 200
        elif not taker_achieved_chelem and self._chelem_announced:
            contract_value -= 200

        if is_petit_played_in_round:
            to_add = (10 if not is_taker_win_round else -10) * multiplier
            if taker_points > victory_threshold:
                to_add *= -1
            contract_value += to_add

        rewards = [3 * contract_value, -contract_value, -contract_value, -contract_value]
        rewards = self._update_rewards_with_poignee(rewards)

        assert np.sum(rewards) == 0
        return rewards

    @staticmethod
    def has_team_achieved_chelem(winners_per_round: List[str], is_excuse_played_in_round: bool, team: str) -> bool:
        winners_per_round = np.array(winners_per_round)
        team_won_all = np.all(winners_per_round == team)
        team_won_all_but_last = np.all(winners_per_round[:-1] == team)
        is_chelem_achieved = team_won_all or (team_won_all_but_last and is_excuse_played_in_round)
        return is_chelem_achieved

    def _update_rewards_with_poignee(self, rewards: List[float]) -> List[float]:
        rewards = copy.copy(rewards)
        for player, announcements_for_player in enumerate(self._announcements):
            for announcement in announcements_for_player:
                if isinstance(announcement, PoigneeAnnouncement):
                    poignee_size_to_bonus = {
                        PoigneeAnnouncement.SIMPLE_POIGNEE_SIZE: 20,
                        PoigneeAnnouncement.DOUBLE_POIGNEE_SIZE: 30,
                        PoigneeAnnouncement.TRIPLE_POIGNEE_SIZE: 40
                    }
                    bonus = poignee_size_to_bonus[len(announcement)]
                    is_player_won = rewards[player] > 0
                    if player == 0:
                        if is_player_won:
                            rewards[0] += 3 * bonus
                            rewards[1] -= bonus
                            rewards[2] -= bonus
                            rewards[3] -= bonus
                        else:
                            rewards[0] -= 3 * bonus
                            rewards[1] += bonus
                            rewards[2] += bonus
                            rewards[3] += bonus
                    else:
                        is_taker_won = rewards[0] > 0
                        if is_taker_won:
                            for i in range(1, len(rewards)):
                                if i == player:
                                    rewards[i] += 2 * bonus
                                else:
                                    rewards[i] -= bonus
                        else:
                            rewards[0] -= 3 * bonus
                            rewards[1] += bonus
                            rewards[2] += bonus
                            rewards[3] += bonus
        return rewards

    def _solve_round(self) -> List[float]:
        starting_player = self._get_next_player()
        winning_card_index = FrenchTarotEnvironment._get_winning_card_index(self._played_cards_in_round)
        play_order = np.arange(starting_player, starting_player + self.n_players) % self.n_players
        winner = play_order[winning_card_index]
        reward_for_winner = get_card_set_point(self._played_cards_in_round)
        rewards = []
        if winner == 0:  # if winner is taking player
            rewards.append(reward_for_winner)
            rewards.extend([0] * (self.n_players - 1))
            self._winners_per_round.append("taker")
        else:
            rewards.append(0)
            rewards.extend([reward_for_winner] * (self.n_players - 1))
            self._winners_per_round.append("opponents")
        self._past_rounds.append(Round(self._played_cards_in_round, starting_player))

        won_cards = self._played_cards_in_round.copy()
        if Card.EXCUSE in self._played_cards_in_round:
            excuse_owner = play_order[self._played_cards_in_round.index(Card.EXCUSE)]
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
        self.current_player = winner
        self._played_cards_in_round = []
        return rewards

    @staticmethod
    def _get_winning_card_index(played_cards: List[Card]) -> int:
        asked_color = _retrieve_asked_color(played_cards)
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
        return int(np.argmax(card_strengths))

    def _announce(self, action: List[Announcement]) -> Tuple[float, bool, any]:
        if not isinstance(action, list):
            raise ValueError("Input should be list")
        for announcement in action:
            if not isinstance(announcement, Announcement):
                raise ValueError("Invalid action type")
            elif isinstance(announcement, PoigneeAnnouncement):
                current_player_hand = self._hand_per_player[self.current_player]
                n_cards = len(announcement)
                if count_trumps_and_excuse(announcement.revealed_cards) != n_cards:
                    raise ValueError("Invalid cards in poignee")
                if n_cards != PoigneeAnnouncement.SIMPLE_POIGNEE_SIZE \
                        and n_cards != PoigneeAnnouncement.DOUBLE_POIGNEE_SIZE \
                        and n_cards != PoigneeAnnouncement.TRIPLE_POIGNEE_SIZE:
                    raise ValueError("Invalid poignee size")
                n_trumps_in_hand = count_trumps_and_excuse(current_player_hand)
                if Card.EXCUSE in announcement.revealed_cards and n_trumps_in_hand != n_cards:
                    raise ValueError("Excuse can be revealed only if player does not have any other trumps")
                elif count_trumps_and_excuse(announcement.revealed_cards) != n_cards:
                    raise ValueError("Revealed cards should be only trumps or excuse")
                elif np.any([card not in current_player_hand for card in announcement.revealed_cards]):
                    raise ValueError("Revealed card not owned by player")
            elif isinstance(announcement, ChelemAnnouncement):
                if len(self._announcements) > 0:
                    raise ValueError("Only taker can announce chelem")
                self._chelem_announced = True

        if np.sum([isinstance(announcement, list) for announcement in action]) > 1:
            raise ValueError("Player tried to announcement more than 1 poignees")

        self._announcements.append(action)
        if len(self._announcements) == self.n_players:
            self._game_phase = GamePhase.CARD
            self.current_player = 0 if self._chelem_announced else self._starting_player
        else:
            self.current_player = self._get_next_player()

        reward = 0
        done = False
        info = None
        return reward, done, info

    def _make_dog(self, dog: List[Card]) -> Tuple[float, bool, any]:
        taking_player_hand = self._hand_per_player[0]  # At this point, taking player is always player 0
        if type(dog) != list:
            raise ValueError("Wrong type for 'action'")
        if len(set(dog)) != len(dog):
            raise ValueError("Duplicated cards in dog")
        if np.any(["king" in card.value for card in dog]):
            raise ValueError("There should be no king in dog")
        if np.any([is_oudler(card) for card in dog]):
            raise ValueError("There should be no oudler in dog")
        if np.any([card not in taking_player_hand for card in dog]):
            raise ValueError("Card in dog not in taking player's hand")
        if len(dog) != self._n_cards_in_dog:
            raise ValueError("Wrong number of cards in dog")

        n_trumps_in_dog = np.sum(["trump" in card.value for card in dog])
        if n_trumps_in_dog > 0:
            n_trumps_in_taking_player_hand = count_trumps_and_excuse(taking_player_hand)
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
        self.current_player = 0
        index_to_keep_in_hand = [card not in dog for card in taking_player_hand]
        self._hand_per_player[0] = list(np.array(taking_player_hand)[index_to_keep_in_hand])
        reward = get_card_set_point(dog)
        self._made_dog = dog
        done = False
        info = None
        return reward, done, info

    def _bid(self, action: Bid) -> Tuple[float, bool, any]:
        if type(action) != Bid:
            raise ValueError("Wrong type for 'action'")

        if action != Bid.PASS and action < get_minimum_allowed_bid(self._bid_per_player):
            raise ValueError("Action is not pass and is lower than highest bid")

        self._bid_per_player.append(action)
        self.current_player = len(self._bid_per_player)
        reward = 0
        if len(self._bid_per_player) == self.n_players:
            done = np.all(np.array(self._bid_per_player) == Bid.PASS)
            taking_player = int(np.argmax(self._bid_per_player))
            original_player_ids = np.arange(taking_player, taking_player + self.n_players) % self.n_players
            self._original_player_ids = list(original_player_ids)
            self._bid_per_player = rotate_list(self._bid_per_player, -taking_player)
            assert np.argmax(self._bid_per_player) == 0
            self._hand_per_player = rotate_list(self._hand_per_player, -taking_player)
            self._starting_player = -taking_player % self.n_players
            self.taking_player_original_id = taking_player
            if np.max(self._bid_per_player) <= Bid.GARDE:
                self._hand_per_player[0] = self._hand_per_player[0] + self._original_dog
                self._game_phase = GamePhase.DOG
                self.current_player = self._starting_player
            else:
                self._game_phase = GamePhase.ANNOUNCEMENTS
                self.current_player = 0  # taker makes announcements first
        else:
            done = False
        info = None

        if done:
            reward = [0] * self.n_players

        return reward, done, info

    def reset(self) -> Observation:
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
        self._chelem_announced = False
        self.current_player = 0
        self._played_cards_in_round = []
        self._past_rounds = []
        self._won_cards_per_teams = {"taker": [], "opponents": []}
        self._bonus_points_per_teams = {"taker": 0., "opponents": 0.}
        self._made_dog = None
        self._winners_per_round = []
        self._original_player_ids = []

        return self._get_observation()

    def _deal(self, deck: List[Card]):
        if len(deck) != len(CARDS):
            raise ValueError("Deck has wrong number of cards")
        self._hand_per_player = [
            deck[:self._n_cards_per_player],
            deck[self._n_cards_per_player:2 * self._n_cards_per_player],
            deck[2 * self._n_cards_per_player:3 * self._n_cards_per_player],
            deck[3 * self._n_cards_per_player:4 * self._n_cards_per_player],
        ]
        self._original_dog = deck[-self._n_cards_in_dog:]
        for hand in self._hand_per_player:
            if Card.TRUMP_1 in hand and count_trumps_and_excuse(hand) == 1:
                raise RuntimeError("'Petit sec'. Deal again.")

    def _get_observation(self) -> Observation:
        current_hand = self._hand_per_player[self.current_player]
        if self._game_phase == GamePhase.BID:
            observation = BidPhaseObservation(self._game_phase, self._bid_per_player, self.current_player, current_hand)
        else:
            original_dog = self._original_dog if np.max(self._bid_per_player) <= Bid.GARDE else "unrevealed"
            if self._game_phase == GamePhase.DOG:
                observation = DogPhaseObservation(self._game_phase, self._bid_per_player, self.current_player,
                                                  self._hand_per_player[0], original_dog, self._original_player_ids)
            elif self._game_phase == GamePhase.ANNOUNCEMENTS:
                observation = AnnouncementPhaseObservation(self._game_phase, self._bid_per_player, self.current_player,
                                                           current_hand, original_dog, self._original_player_ids,
                                                           self._revealed_cards_in_dog, self._announcements)
            elif self._game_phase == GamePhase.CARD:
                observation = CardPhaseObservation(self._game_phase, self._bid_per_player, self.current_player,
                                                   current_hand, original_dog, self._original_player_ids,
                                                   self._revealed_cards_in_dog, self._announcements,
                                                   self._played_cards_in_round, self._past_rounds)
            else:
                raise RuntimeError("Unknown game phase")

        return copy.deepcopy(observation)

    def render(self, mode="human", close=False):
        raise NotImplementedError()


def _retrieve_asked_color(played_cards: List[Card]) -> str:
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


def get_trumps_and_excuse(cards: List[Card]) -> List[Card]:
    output_as_list = isinstance(cards, list)
    cards = np.array(cards)
    rval = cards[np.array(["trump" in card.value or card.value == "excuse" for card in cards])]
    if output_as_list:
        rval = list(rval)
    return rval


def is_oudler(card: Card) -> bool:
    return card == Card.TRUMP_1 or card == Card.TRUMP_21 or card == Card.EXCUSE


def get_card_point(card: Card) -> float:
    if is_oudler(card) or "king" in card.value:
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


def get_card_set_point(card_list: List[Card]) -> float:
    return float(np.sum([get_card_point(card) for card in card_list]))
