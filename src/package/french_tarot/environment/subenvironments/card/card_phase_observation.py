from typing import List

from attr import dataclass

from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import Observation


@dataclass
class CardPhaseObservation(Observation):
    played_cards_in_round: List[Card]
