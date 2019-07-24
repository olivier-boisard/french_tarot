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


@pytest.fixture(scope="module")
def random_agent():
    return RandomPlayer()


@pytest.mark.repeat(10)
def test_bid_phase(random_agent):
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    observation = environment.step(random_agent.get_action(observation))[0]
    observation = environment.step(random_agent.get_action(observation))[0]
    observation = environment.step(random_agent.get_action(observation))[0]
    environment.step(random_agent.get_action(observation))
    # No assertion needed here, we just need to make sure agent always provided valid actions, meaning there will
    # be no exceptions raised.
