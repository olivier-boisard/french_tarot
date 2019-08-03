import numpy as np
import pytest

from random_agent import RandomPlayer
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


@pytest.fixture(scope="module")
def environment():
    return FrenchTarotEnvironment()


@pytest.mark.repeat(10)
def test_play_game(random_agent, environment):
    observation = environment.reset()
    done = False
    cnt = 0
    while not done:
        observation, _, done, _ = environment.step(random_agent.get_action(observation))
        cnt += 1
        if cnt >= 1000:
            raise RuntimeError("Infinite loop")

    # No assert needed here, the code just needs to run without raising exceptions
