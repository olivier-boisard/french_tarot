from french_tarot.agents.common import core, CoreCardNeuralNet
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.common import Bid
from french_tarot.environment.environment import FrenchTarotEnvironment
from french_tarot.environment.observations import Observation


def test_dog_phase_observation_encoder():
    observation = _prepare_environment()
    state = core(observation.hand)

    assert state.shape[0] == 78
    assert state.sum() == 24


def test_create_dog_phase_player():
    player = DogPhaseAgent(CoreCardNeuralNet(), device="cpu")
    observation = _prepare_environment()
    action = player.get_action(observation)
    assert isinstance(action, list)
    assert len(action) == 6


def _prepare_environment() -> Observation:
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation = environment.step(Bid.PASS)[0]
    return observation
