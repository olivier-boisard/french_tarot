import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer.managers.manager import Manager


@pytest.fixture
def manager():
    return Manager()


@pytest.fixture(scope="module")
def environment():
    return FrenchTarotEnvironment()


def create_teardown_func(*threads):
    def teardown():
        for thread in threads:
            thread.stop()

    return teardown


@pytest.fixture
def random_agent():
    return RandomPlayer()
