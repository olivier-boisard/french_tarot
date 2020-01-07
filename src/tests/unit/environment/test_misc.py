from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import get_card_point


def test_point_count_spades_1():
    assert get_card_point(Card.SPADES_1) == 0.5


def test_point_count_spades_king():
    assert get_card_point(Card.SPADES_KING) == 4.5


def test_point_count_card_set():
    assert compute_card_set_points(CARDS) == 91
