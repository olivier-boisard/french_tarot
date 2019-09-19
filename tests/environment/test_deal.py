import copy

import pytest

from french_tarot.environment.core import CARDS, Card
from french_tarot.environment.french_tarot import FrenchTarotEnvironment


def test_deal():
    card_list = copy.copy(CARDS)
    card_list[0] = Card.TRUMP_1
    card_list[-21] = Card.SPADES_1
    environment = FrenchTarotEnvironment()
    environment.reset()
    with pytest.raises(RuntimeError):
        environment._deal(card_list)
