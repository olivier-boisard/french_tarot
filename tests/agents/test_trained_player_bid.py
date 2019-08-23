import pytest

from agents.common import card_set_encoder
from agents.trained_player_bid import BidPhaseAgent
from environment import FrenchTarotEnvironment, Bid, GamePhase


def test_bid_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = card_set_encoder(observation["hand"])

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
