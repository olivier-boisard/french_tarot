import pytest

from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer import Manager


@pytest.fixture
def manager():
    return Manager()


@pytest.fixture(scope="module")
def environment():
    return FrenchTarotEnvironment()
