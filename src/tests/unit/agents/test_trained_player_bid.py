from torch import nn

from french_tarot.agents.bid_phase_agent import BidPhaseAgent
from french_tarot.agents.encoding import encode_cards_as_tensor
from french_tarot.environment.core.bid import Bid
from french_tarot.environment.french_tarot_environment import FrenchTarotEnvironment


def test_bid_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = encode_cards_as_tensor(observation.player.hand)

    assert state.shape[0] == 78
    assert state.sum() == 18


def test_create_bid_phase_player():
    player = BidPhaseAgent(nn.Linear(78, 4))
    observation = FrenchTarotEnvironment().reset()
    action = player.get_action(observation)
    assert isinstance(action, Bid)
