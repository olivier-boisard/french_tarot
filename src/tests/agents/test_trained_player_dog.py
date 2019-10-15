from french_tarot.agents.common import encode_cards, CoreCardNeuralNet
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.core import Bid
from french_tarot.environment.french_tarot import FrenchTarotEnvironment


def test_dog_phase_observation_encoder():
    observation = _prepare_environment()
    state = encode_cards(observation.player.hand)

    assert state.shape[0] == 78
    assert state.sum() == 24


def test_create_dog_phase_player():
    player = DogPhaseAgent(DogPhaseAgent.create_dqn(CoreCardNeuralNet()))
    observation = _prepare_environment()
    action = player.get_action(observation)
    assert isinstance(action, list)
    assert len(action) == 6


def _prepare_environment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation = environment.step(Bid.PASS)[0]
    return observation
