from french_tarot.agents.common import core, BaseCardNeuralNet
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.environment.common import Bid
from french_tarot.environment.environment import FrenchTarotEnvironment


def test_bid_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = core(observation.hand)

    assert state.shape[0] == 78
    assert state.sum() == 18


def test_create_bid_phase_player():
    player = BidPhaseAgent(BaseCardNeuralNet(), device="cpu")
    observation = FrenchTarotEnvironment().reset()
    action = player.get_action(observation)
    assert isinstance(action, Bid)
