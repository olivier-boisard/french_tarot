from typing import List

from attr import dataclass

from french_tarot.environment.core import Observation, Card


@dataclass
class CardPhaseObservation(Observation):
    played_cards_in_round: List[Card]
