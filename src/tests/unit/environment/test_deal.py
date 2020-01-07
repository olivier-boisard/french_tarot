import copy

import pytest

from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import CARDS
from french_tarot.environment.french_tarot_environment import FrenchTarotEnvironment
from french_tarot.exceptions import FrenchTarotException


def test_deal_petit_sec():
    card_list = copy.copy(CARDS)
    card_list[0] = Card.TRUMP_1
    card_list[-21] = Card.SPADES_1
    environment = FrenchTarotEnvironment()
    with pytest.raises(FrenchTarotException):
        environment.reset(card_list)
