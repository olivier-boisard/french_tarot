from abc import ABC, abstractmethod
from typing import Tuple

from french_tarot.environment.core import Bid, Observation


class SubEnvironment(ABC):

    @abstractmethod
    def reset(self) -> Observation:
        pass

    @abstractmethod
    def step(self, action: Bid) -> Tuple[Observation, float, bool, any]:
        pass

    @property
    @abstractmethod
    def game_is_done(self):
        pass
