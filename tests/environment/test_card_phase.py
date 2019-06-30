import pytest

from environment import FrenchTarotEnvironment, Bid, Card, get_card_set_point


def setup_envionment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    return environment


def test_start_valid_card():
    environment = setup_envionment()
    played_card = environment._hand_per_player[0][0]
    observation, reward, done, _ = environment.step(played_card)
    assert observation["played_cards"] == [played_card]
    assert reward == 0
    assert not done


def test_start_invalid_card():
    environment = setup_envionment()
    played_card = environment._hand_per_player[1][0]
    with pytest.raises(ValueError):
        environment.step(played_card)


def test_play_complete_round_valid_last_player_team_wins():
    environment = setup_envionment()
    environment.step(Card.HEART_1)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    assert observation["won_cards_per_team"]["taker"] == []
    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["won_cards_per_team"]["opponents"] == expected_values
    assert observation["plis"] == [expected_values]
    assert environment._current_player == 1
    assert reward == get_card_set_point(expected_values)  # last player's team won this round
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_envionment()
    environment._current_player = 1
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    observation, reward, done, _ = environment.step(Card.HEART_1)

    assert observation["won_cards_per_team"]["taker"] == []
    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["won_cards_per_team"]["opponents"] == expected_values
    assert observation["plis"] == [expected_values]
    assert environment._current_player == 1
    assert reward == 0  # last player's team lost this round
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_envionment()
    environment._current_player = 1
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    observation, reward, done, _ = environment.step(Card.HEART_1)

    assert observation["won_cards_per_team"]["taker"] == []
    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["won_cards_per_team"]["opponents"] == expected_values
    assert observation["plis"] == [expected_values]
    assert environment._current_player == 1
    assert reward == 0  # last player's team lost this round
    assert not done


def test_play_excuse_in_round():
    raise NotImplementedError()


def test_play_excuse_first():
    raise NotImplementedError()


def test_play_trump_valid():
    raise NotImplementedError()


def test_play_trump_invalid():
    raise NotImplementedError()


def test_play_trump_above_trump():
    raise NotImplementedError()


def test_play_trump_below_trump_unallowed():
    raise NotImplementedError()


def test_play_trump_below_trump_allowed():
    raise NotImplementedError()


def test_pee_allowed():
    raise NotImplementedError()


def test_pee_not_allowed():
    raise NotImplementedError()


def test_play_card_not_in_hand():
    raise NotImplementedError()


def test_play_last_cards():
    raise NotImplementedError()
