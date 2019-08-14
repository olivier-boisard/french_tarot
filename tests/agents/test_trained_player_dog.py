import numpy as np
import pytest

from agents.common import card_set_encoder
from agents.random_agent import RandomPlayer
from environment import FrenchTarotEnvironment, GamePhase


def test_dog_phase_observation_encoder():
    observation = _prepare_environment()
    state = card_set_encoder(observation)

    assert state.shape[0] == 78
    assert state.sum() == 24


def test_create_bid_phase_player():
    player = DogPhaseAgent(device="cpu")
    observation = _prepare_environment(player)
    action = player.get_action(observation)
    assert isinstance(action, list)
    assert np.sum(action) == 6


def _prepare_environment(player=None):
    if player is None:
        player = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    while observation["game_phase"] != GamePhase.DOG:
        observation = environment.step(player.get_action(observation))[0]
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
