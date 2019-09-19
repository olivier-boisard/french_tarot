from abc import ABC, abstractmethod
from typing import Tuple

from french_tarot.environment.core import Bid


class SubEnvironment(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: Bid) -> Tuple[any, float, bool, any]:
        pass

    @property
    @abstractmethod
    def game_is_done(self):
        pass
