import pytest

from environment import FrenchTarotEnvironment, Bid, Card, GamePhase


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
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS


def test_make_dog_with_duplicated_card(environment):
    dog = [Card.SPADES_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_king(environment):
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_KING]
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_oudler(environment):
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.EXCUSE]
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_trump_invalid(environment):
    dog = [Card.TRUMP_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_trump_valid():
    dog = [Card.TRUMP_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = FrenchTarotEnvironment()
    environment._deal(list(Card))
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward == 0
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS
    assert observation["revealed_cards_in_dog"] == [Card.TRUMP_1]
