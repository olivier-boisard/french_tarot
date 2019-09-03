import pytest

from environment import FrenchTarotEnvironment, Card


def test_deal():
    # noinspection PyTypeChecker
    card_list = list(Card)
    card_list[0] = Card.TRUMP_1
    card_list[-21] = Card.SPADES_1
    environment = FrenchTarotEnvironment()
    environment.reset()
    with pytest.raises(RuntimeError):
        environment._deal(card_list)
