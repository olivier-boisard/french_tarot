from environment import Card, get_card_point, get_card_set_point


def test_point_count_spades_1():
    assert get_card_point(Card.SPADES_1) == 0.5


def test_point_count_spades_king():
    assert get_card_point(Card.SPADES_KING) == 4.5


def test_point_count_card_set():
    assert get_card_set_point(list(Card)) == 91
