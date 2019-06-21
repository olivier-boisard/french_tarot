import pytest

from environment import FrenchTarotEnvironment, Bid, Card, GamePhase


def setup_environment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    return environment


def test_make_dog():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward == 0
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS


def test_make_dog_with_duplicated_card():
    dog = [Card.SPADES_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_king():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_KING]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_oudler():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.EXCUSE]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_trump_invalid():
    dog = [Card.TRUMP_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_trump_valid():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    dog = [Card.DIAMOND_QUEEN, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6]
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward == 0
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS
    assert observation["revealed_cards_in_dog"] == [Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5,
                                                    Card.TRUMP_6]


def test_dog_with_card_not_in_players_hand():
    raise NotImplementedError()


def test_dog_has_wrong_number_of_cards():
    raise NotImplementedError()
