from environment import FrenchTarotEnvironment
from trained_player import bid_phase_observation_encoder


def test_bid_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = bid_phase_observation_encoder(observation)
    assert state.shape[0] == 78
    assert state.sum() == 18
