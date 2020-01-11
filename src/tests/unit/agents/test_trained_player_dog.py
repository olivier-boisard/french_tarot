import pytest
from torch import nn

from french_tarot.agents.dog_phase_agent import DogPhaseAgent
from french_tarot.agents.encoding import encode_cards_as_tensor
from french_tarot.environment.core.bid import Bid
from french_tarot.environment.french_tarot_environment import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.dog.dog_phase_observation import DogPhaseObservation


@pytest.fixture
def dog_phase_agent():
    return DogPhaseAgent(nn.Linear(156, 78))


def test_dog_phase_observation_encoder():
    observation = _prepare_environment()
    state = encode_cards_as_tensor(observation.player.hand)

    assert state.shape[0] == 78
    assert state.sum() == 24


@pytest.mark.skip
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
