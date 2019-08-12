import numpy as np
import pytest

from environment import FrenchTarotEnvironment, GamePhase


def test_dog_phase_observation_encoder():
    observation = FrenchTarotEnvironment().reset()
    state = dog_phase_observation_encoder(observation)

    assert state.shape[0] == 78 * 2
    assert state.sum() == 78 + 6


def test_create_bid_phase_player():
    player = DogPhaseAgent(device="cpu")
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    while observation["game_phase"] != GamePhase.DOG:
        observation = environment.step(player.get_action(observation))
    action = player.get_action(observation)
    assert isinstance(action, list)
    assert np.sum(action) == 6


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
