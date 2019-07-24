import numpy as np
import pytest

from agent import RandomPlayer
from environment import FrenchTarotEnvironment, Bid


def test_instantiate_random_player():
    random_agent = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    action = random_agent.get_action(observation)
    assert isinstance(action, Bid)


def test_randomness_when_bidding():
    random_agent = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    actions = [random_agent.get_action(observation) for _ in range(10)]
    assert len(np.unique(actions)) > 1


@pytest.mark.repeat(10)  # TODO does this work?
def test_bid_phase():
    raise NotImplementedError()
