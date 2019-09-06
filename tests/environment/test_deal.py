import copy

import pytest

from environment import FrenchTarotEnvironment, Card, CARDS


def test_deal():
    card_list = copy.copy(CARDS)
    card_list[0] = Card.TRUMP_1
    card_list[-21] = Card.SPADES_1
    environment = FrenchTarotEnvironment()
    environment.reset()
    with pytest.raises(RuntimeError):
        environment._deal(card_list)
