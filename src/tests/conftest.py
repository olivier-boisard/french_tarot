import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.french_tarot import FrenchTarotEnvironment


@pytest.fixture(scope="module")
def environment():
    return FrenchTarotEnvironment()


@pytest.fixture
def random_agent():
    return RandomPlayer()
