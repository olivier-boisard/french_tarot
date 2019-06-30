import pytest

from environment import FrenchTarotEnvironment, Bid, Card, get_card_set_point


def setup_environment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step([])
    environment.step([])
    environment.step([])
    environment.step([])
    return environment


def test_start_valid_card():
    environment = setup_environment()
    played_card = environment._hand_per_player[0][0]
    observation, reward, done, _ = environment.step(played_card)
    assert observation["played_cards"] == [played_card]
    assert reward is None
    assert not done


def test_start_invalid_card():
    environment = setup_environment()
    played_card = environment._hand_per_player[1][0]
    with pytest.raises(ValueError):
        environment.step(played_card)


def test_play_complete_round_valid_last_player_team_wins():
    environment = setup_environment()
    environment.step(Card.HEART_1)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"] == [expected_values]
    assert environment._current_player == 1
    assert reward[0] == 0
    assert reward[1] == get_card_set_point(expected_values)
    assert reward[2] == get_card_set_point(expected_values)
    assert reward[3] == get_card_set_point(expected_values)
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_environment()
    environment._current_player = 1
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    observation, reward, done, _ = environment.step(Card.HEART_1)

    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"] == [expected_values]
    assert environment._current_player == 1
    assert reward == 0  # last player's team lost this round
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_environment()
    starting_player = 1
    environment._current_player = starting_player
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    observation, reward, done, _ = environment.step(Card.HEART_1)

    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"] == [{"played_cards": expected_values, "starting_player": starting_player}]
    assert environment._current_player == 1
    assert reward == 0  # last player's team lost this round
    assert not done


def test_play_excuse_in_round():
    environment = setup_environment()
    environment._current_player = 2
    observation_0 = environment.step(Card.HEART_4)[0]
    observation_1 = environment.step(Card.HEART_2)[0]
    observation_2 = environment.step(Card.EXCUSE)[0]
    observation_3, reward, done, _ = environment.step(Card.HEART_KING)

    assert observation_0["played_card"] == []
    assert observation_1["played_card"] == [Card.HEART_4]
    assert observation_2["played_card"] == [Card.HEART_4, Card.HEART_2]
    assert observation_3["played_card"] == [Card.HEART_4, Card.HEART_2, Card.EXCUSE]

    expected_values = [Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation_3["plis"] == [[Card.HEART_4, Card.HEART_2, Card.EXCUSE, Card.HEART_KING]]
    assert environment._current_player == 1
    assert reward == get_card_set_point(expected_values)  # last player's team won this round
    assert not done


def test_play_excuse_first():
    environment = setup_environment()
    environment.step(Card.EXCUSE)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    expected_values = [Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"] == [Card.EXCUSE, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert environment._current_player == 1
    assert reward == get_card_set_point(expected_values)  # last player's team won this round
    assert not done


def test_play_trump_invalid():
    environment = setup_environment()
    environment._hand_per_player = [[Card.SPADES_1, Card.SPADES_2], [Card.SPADES_3, Card.TRUMP_1]]
    environment.step(Card.SPADES_1)
    with pytest.raises(ValueError):
        environment.step(Card.TRUMP_1)


def test_play_trump_below_trump_unallowed():
    environment = setup_environment()
    environment._hand_per_player = [[Card.TRUMP_10, Card.TRUMP_11], [Card.TRUMP_1, Card.TRUMP_12]]
    environment.step(Card.TRUMP_10)
    with pytest.raises(ValueError):
        environment.step(Card.TRUMP_1)


def test_pee_not_allowed():
    environment = setup_environment()
    environment._hand_per_player = [[Card.SPADES_1, Card.SPADES_2], [Card.HEART_4, Card.SPADES_3]]
    environment.step(Card.SPADES_1)
    with pytest.raises(ValueError):
        environment.step(Card.HEART_4)


def test_play_card_not_in_hand():
    environment = setup_environment()
    environment._hand_per_player = [[Card.SPADES_1]]
    with pytest.raises(ValueError):
        environment.step(Card.HEART_4)


def test_play_complete_game():
    environment = setup_environment()
    raise NotImplementedError()
