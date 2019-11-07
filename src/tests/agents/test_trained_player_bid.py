from torch import nn

from french_tarot.agents.encoding import encode_cards
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.environment.core import Bid
from french_tarot.environment.french_tarot import FrenchTarotEnvironment


def test_bid_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = encode_cards(observation.player.hand)

    assert state.shape[0] == 78
    assert state.sum() == 18


def test_create_bid_phase_player():
    player = BidPhaseAgent(nn.Linear(78, 4))
    observation = FrenchTarotEnvironment().reset()
    action = player.get_action(observation)
    assert isinstance(action, Bid)
