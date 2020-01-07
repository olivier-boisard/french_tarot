from typing import List

from attr import dataclass

from french_tarot.environment.core import Card


@dataclass
class Round:
    starting_player_id: int
    played_cards: List[Card]
