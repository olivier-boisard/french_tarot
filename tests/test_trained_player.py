import pytest

from environment import FrenchTarotEnvironment, Bid, GamePhase
from trained_player import bid_phase_observation_encoder, BidPhaseAgent


def test_bid_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = bid_phase_observation_encoder(observation)

    assert state.shape[0] == 78
    assert state.sum() == 18


def test_create_bid_phase_player():
    player = BidPhaseAgent(device="cpu")
    observation = FrenchTarotEnvironment().reset()
    action = player.get_action(observation)
    assert isinstance(action, Bid)


def test_create_bid_phase_player_wrong_phase():
    player = BidPhaseAgent(device="cpu")
    observation = FrenchTarotEnvironment().reset()
    observation["game_phase"] = GamePhase.DOG
    with pytest.raises(ValueError):
        player.get_action(observation)
