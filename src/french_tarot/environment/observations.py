from abc import ABC
from typing import List

from french_tarot.environment.common import Card, GamePhase, Bid, Announcement


class Observation(ABC):

    def __init__(self, game_phase: GamePhase, bid_per_player: List[Bid], current_player_id: int, hand: List[Card]):
        self.game_phase = game_phase
        self.bid_per_player = bid_per_player
        self.current_player_id = current_player_id
        self.hand = hand


class _AfterBidPhaseObservation(Observation):
    def __init__(
            self,
            game_phase: GamePhase,
            bid_per_player: List[Bid],
            current_player_id: int,
            hand: List[Card],
            original_dog: List[Card],
            original_player_ids: List[int]
    ):
        super(_AfterBidPhaseObservation, self).__init__(game_phase, bid_per_player, current_player_id, hand)
        self.original_dog = original_dog
        self.original_player_ids = original_player_ids


class _AfterDogPhaseObservation(_AfterBidPhaseObservation):
    def __init__(
            self,
            game_phase: GamePhase,
            bid_per_player: List[Bid],
            current_player_id: int,
            hand: List[Card],
            original_dog: List[Card],
            original_player_ids: List[int],
            revealed_cards_in_new_dog: List[Card],
            announcements: List[Announcement]
    ):
        super(_AfterBidPhaseObservation, self).__init__(game_phase, bid_per_player, current_player_id, hand)
        self.original_dog = original_dog
        self.original_player_ids = original_player_ids
        self.revealed_cards_in_new_dog = revealed_cards_in_new_dog
        self.announcements = announcements


class Round:

    def __init__(self, played_cards: List[Card], starting_player_id: int):
        self.played_cards = played_cards
        self.starting_player_id = starting_player_id


class BidPhaseObservation(Observation):
    pass


class DogPhaseObservation(_AfterBidPhaseObservation):
    pass


class AnnouncementPhaseObservation(_AfterDogPhaseObservation):
    pass


class CardPhaseObservation(_AfterDogPhaseObservation):
    def __init__(
            self,
            game_phase: GamePhase,
            bid_per_player: List[Bid],
            current_player_id: int,
            hand: List[Card],
            original_dog: List[Card],
            original_player_ids: List[int],
            revealed_cards_in_new_dog: List[Card],
            announcements: List[Announcement],
            played_cards_in_round: List[Card],
            past_rounds: List[Round]
    ):
        super(CardPhaseObservation, self).__init__(game_phase, bid_per_player, current_player_id, hand,
                                                   original_dog, original_player_ids,
                                                   revealed_cards_in_new_dog, announcements)
        self.played_cards_in_round = played_cards_in_round
        self.past_rounds = past_rounds
