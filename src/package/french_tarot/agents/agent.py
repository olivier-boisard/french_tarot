from abc import ABC, abstractmethod
from dataclasses import dataclass

from french_tarot.environment.core.core import Observation


class Agent(ABC):

    @abstractmethod
    def get_action(self, observation: Observation) -> 'ActionWithProbability':
        pass


@dataclass
class ActionWithProbability:
    action: int
    probability: float
