import pytest

from french_tarot.agents.encoding import encode_cards
from french_tarot.agents.neural_net import CoreCardNeuralNet
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.core import Bid
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


@pytest.fixture
def dog_phase_agent():
    return DogPhaseAgent(DogPhaseAgent.create_dqn(CoreCardNeuralNet()))


def test_dog_phase_observation_encoder():
    observation = _prepare_environment()
    state = encode_cards(observation.player.hand)

    assert state.shape[0] == 78
    assert state.sum() == 24


def test_create_dog_phase_player(dog_phase_agent):
    observation = _prepare_environment()
    action = dog_phase_agent.get_action(observation)
    assert isinstance(action, list)
    assert len(action) == 6


def test_random_action(dog_phase_agent):
    observation = _prepare_environment()
    action = dog_phase_agent.get_random_action(observation)
    assert len(action) == observation.dog_size


def _prepare_environment() -> DogPhaseObservation:
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    return environment.step(Bid.PASS)[0]
