import numpy as np
import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.core import Bid
from french_tarot.environment.french_tarot import FrenchTarotEnvironment


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
