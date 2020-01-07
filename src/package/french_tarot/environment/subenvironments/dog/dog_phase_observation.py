from attr import dataclass

from french_tarot.environment.core import Observation


@dataclass
class DogPhaseObservation(Observation):
    dog_size: int
