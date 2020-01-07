from typing import List

from attr import dataclass

from french_tarot.environment.core import Observation, Bid


@dataclass
class BidPhaseObservation(Observation):
    bid_per_player: List[Bid]
