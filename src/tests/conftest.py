import pytest

from french_tarot.observer import Manager


@pytest.fixture
def manager():
    return Manager()
