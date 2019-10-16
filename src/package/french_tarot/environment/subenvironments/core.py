from abc import ABC, abstractmethod
from typing import Tuple, Union, List

from french_tarot.environment.core import Observation


class SubEnvironment(ABC):

    @abstractmethod
    def reset(self) -> Observation:
        pass

    @abstractmethod
    def step(self, action: any) -> Tuple[Observation, Union[float, List[float]], bool, any]:
        pass

    @property
    @abstractmethod
    def game_is_done(self):
        pass
