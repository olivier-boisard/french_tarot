from attr import dataclass

from french_tarot.environment.core.core import Observation


@dataclass
class DogPhaseObservation(Observation):
    dog_size: int
