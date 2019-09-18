from typing import List

from attr import dataclass

from french_tarot.environment.common import Card, Bid


@dataclass
class BidPhaseObservation:
    hand: List[Card]
    bid_per_player: List[Bid]


@dataclass
class DogPhaseObservation:
    hand: List[Card]
    dog_size: int


@dataclass
class AnnouncementPhaseObservation:
    current_player_id: int
    hand: List[Card]


@dataclass
class CardPhaseObservation:
    hand: List[Card]
    played_cards_in_round: List[Card]
