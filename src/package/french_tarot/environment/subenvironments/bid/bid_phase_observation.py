from typing import List

from attr import dataclass

from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.core import Observation


@dataclass
class BidPhaseObservation(Observation):
    bid_per_player: List[Bid]
