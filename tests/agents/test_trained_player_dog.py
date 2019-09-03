import pytest

from agents.common import card_set_encoder
from agents.trained_player_dog import DogPhaseAgent
from environment import FrenchTarotEnvironment, GamePhase, Bid


def test_dog_phase_observation_encoder():
    observation = _prepare_environment()
    state = card_set_encoder(observation["hand"])

    assert state.shape[0] == 78
    assert state.sum() == 24


def test_create_dog_phase_player():
    player = DogPhaseAgent(device="cpu")
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


def test_bid_phase():
    player = DogPhaseAgent(device="cpu")
    observation = FrenchTarotEnvironment().reset()
    observation["game_phase"] = GamePhase.BID
    with pytest.raises(ValueError):
        player.get_action(observation)


def test_announcement_phase():
    player = DogPhaseAgent(device="cpu")
    observation = FrenchTarotEnvironment().reset()
    observation["game_phase"] = GamePhase.ANNOUNCEMENTS
    with pytest.raises(ValueError):
        player.get_action(observation)
