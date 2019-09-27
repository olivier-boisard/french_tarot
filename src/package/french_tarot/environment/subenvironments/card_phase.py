import copy
from typing import List, Tuple

import numpy as np
from attr import dataclass

from french_tarot.agents.common import Round
from french_tarot.environment.core import Card, ChelemAnnouncement, check_card_is_allowed, get_card_set_point, \
    is_oudler, Bid, retrieve_asked_color, PoigneeAnnouncement, Observation
from french_tarot.environment.subenvironments.core import SubEnvironment
from french_tarot.exceptions import FrenchTarotException


@dataclass
class CardPhaseObservation(Observation):
    hand: List[Card]
    played_cards_in_round: List[Card]


class CardPhaseEnvironment(SubEnvironment):

    def __init__(self, hand_per_player, starting_player, made_dog, original_dog, bid_per_player, announcements):
        self._hand_per_player = hand_per_player
        self.current_player = starting_player
        self._made_dog = made_dog
        self._original_dog = original_dog
        self._bid_per_player = bid_per_player
        self._announcements = announcements
        self._played_cards_in_round = []
        self._past_rounds = []
        self._winners_per_round = []
        self._won_cards_per_teams = {"taker": [], "opponents": []}
        self._bonus_points_per_teams = {"taker": 0., "opponents": 0.}

    def reset(self):
        self._played_cards_in_round = []
        self._past_rounds = []
        self._winners_per_round = []
        self._won_cards_per_teams = {"taker": [], "opponents": []}
        self._bonus_points_per_teams = {"taker": 0., "opponents": 0.}
        return self.observation

    @property
    def observation(self):
        return CardPhaseObservation(self.current_hand, self._played_cards_in_round)

    def step(self, card: Card) -> Tuple[any, List[float], bool, any]:
        if not isinstance(card, Card):
            raise FrenchTarotException("Action must be card")
        check_card_is_allowed(card, self._played_cards_in_round, self._hand_per_player[self.current_player])
        self._played_cards_in_round.append(card)

        current_hand = list(np.array(self.current_hand)[np.array(self.current_hand) != card])
        rewards = None
        done = False
        if isinstance(current_hand, Card):
            self._hand_per_player[self.current_player] = list([current_hand])
        else:
            self._hand_per_player[self.current_player] = current_hand

        if len(self._played_cards_in_round) == self.n_players:
            self._past_rounds.append(Round(self.next_player, self._played_cards_in_round))
            is_petit_played_in_round = Card.TRUMP_1 in self._played_cards_in_round
            is_excuse_played_in_round = Card.EXCUSE in self._played_cards_in_round
            rewards = self._solve_round()
            is_taker_win_round = rewards[0] > 0
            if len(self._hand_per_player[0]) == 0:
                rewards = self._compute_win_loss(is_petit_played_in_round, is_excuse_played_in_round,
                                                 is_taker_win_round)
                done = True

        elif len(self._played_cards_in_round) < self.n_players:
            self.current_player = self.next_player
        else:
            raise RuntimeError("Wrong number of played cards")

        info = None
        return self.observation, rewards, done, info

    @property
    def game_is_done(self):
        is_game_done = True
        for hand in self._hand_per_player:
            if len(hand) > 0:
                is_game_done = False
                break
        return is_game_done

    @property
    def chelem_announced(self):
        return np.any([isinstance(announcement, ChelemAnnouncement) for announcement in self._announcements[0]])

    @property
    def n_players(self):
        return len(self._hand_per_player)

    @property
    def current_hand(self):
        return self._hand_per_player[self.current_player]

    def _compute_win_loss(self, is_petit_played_in_round: bool, is_excuse_played_in_round: bool,
                          is_taker_win_round: bool) -> List[float]:
        dog = self._made_dog if len(self._made_dog) > 0 else self._original_dog
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
        # noinspection PyTypeChecker
        diff = abs(victory_threshold - taker_points)
        contract_value = 25 + diff
        bid_to_multiplier_map = {Bid.PETITE: 1, Bid.GARDE: 2, Bid.GARDE_SANS: 4, Bid.GARDE_CONTRE: 6}
        multiplier = bid_to_multiplier_map[np.max(self._bid_per_player)]
        contract_value = int(contract_value * multiplier)
        if taker_points < victory_threshold:
            contract_value *= -1

        winners_per_round = self._winners_per_round
        taker_achieved_chelem = self.has_team_achieved_chelem(winners_per_round, is_excuse_played_in_round, "taker")
        if taker_achieved_chelem:
            contract_value += 400 if self.chelem_announced else 200
        elif self.has_team_achieved_chelem(winners_per_round, is_excuse_played_in_round, "opponents"):
            contract_value -= 200
        elif not taker_achieved_chelem and self.chelem_announced:
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

    @property
    def next_player(self) -> int:
        return (self.current_player + 1) % self.n_players

    @staticmethod
    def _get_winning_card_index(played_cards: List[Card]) -> int:
        asked_color = retrieve_asked_color(played_cards)
        card_strengths = []
        for card in played_cards:
            # TODO use/create utility functions
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
        # noinspection PyTypeChecker
        return int(np.argmax(card_strengths))

    def _solve_round(self) -> List[float]:
        starting_player = self.next_player
        winning_card_index = self._get_winning_card_index(self._played_cards_in_round)
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
                    bonus = announcement.bonus_points()
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
