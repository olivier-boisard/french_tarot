import pytest

from environment import FrenchTarotEnvironment, Bid, Card


@pytest.fixture(scope="module")
def environment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    yield environment


def test_make_dog(environment):
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward == 0
