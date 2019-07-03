import pytest

from environment import FrenchTarotEnvironment, Bid, Card, get_card_set_point, CHELEM


def setup_environment(taker=0):
    environment = FrenchTarotEnvironment()
    environment.reset()
    good = False
    for i in range(environment._n_players):
        if i == taker:
            environment.step(Bid.GARDE_SANS)
            good = True
        else:
            environment.step(Bid.PASS)
    if not good:
        raise ValueError("No taking player")
    environment.step([])
    environment.step([])
    environment.step([])
    observation = environment.step([])[0]
    return environment, observation


def test_start_valid_card():
    environment = setup_environment()[0]
    played_card = environment._hand_per_player[0][0]
    observation, reward, done, _ = environment.step(played_card)
    assert observation["played_cards"] == [played_card]
    assert reward is None
    assert not done


def test_start_invalid_action():
    environment = setup_environment()[0]
    with pytest.raises(ValueError):
        environment.step(CHELEM)


def test_start_invalid_card_not_in_hand():
    environment = setup_environment()[0]
    played_card = environment._hand_per_player[1][0]
    with pytest.raises(ValueError):
        environment.step(played_card)


def test_play_complete_round_valid_last_player_team_wins():
    environment = setup_environment()[0]
    environment.step(Card.HEART_1)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"][0]["played_cards"] == expected_values
    assert environment._current_player == 1
    assert reward[0] == 0
    assert reward[1] == get_card_set_point(expected_values)
    assert reward[2] == get_card_set_point(expected_values)
    assert reward[3] == get_card_set_point(expected_values)
    assert observation["played_cards"] == []
    assert environment._current_player == 1
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_environment()[0]
    starting_player = 1
    environment._current_player = starting_player
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    observation, reward, done, _ = environment.step(Card.HEART_1)

    expected_values = [Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"] == [{"played_cards": expected_values, "starting_player": starting_player}]
    assert environment._current_player == 1
    assert reward[0] == 0
    assert reward[1] == get_card_set_point(expected_values)
    assert reward[2] == get_card_set_point(expected_values)
    assert reward[3] == get_card_set_point(expected_values)
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_environment(taker=1)[0]
    starting_player = 0
    environment._current_player = starting_player
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    observation, reward, done, _ = environment.step(Card.HEART_1)

    expected_values = [Card.HEART_KING, Card.HEART_4, Card.HEART_2, Card.HEART_1]
    assert observation["plis"] == [{"played_cards": expected_values, "starting_player": starting_player}]
    assert environment._current_player == 0
    assert reward[0] == get_card_set_point(expected_values)
    assert reward[1] == 0
    assert reward[2] == 0
    assert reward[3] == 0
    assert not done


def test_play_excuse_in_round():
    environment, observation_0 = setup_environment()
    environment._current_player = 2
    observation_1 = environment.step(Card.HEART_4)[0]
    observation_2 = environment.step(Card.HEART_2)[0]
    observation_3 = environment.step(Card.EXCUSE)[0]
    observation_4, reward, done, _ = environment.step(Card.HEART_KING)

    assert observation_0["played_cards"] == []
    assert observation_1["played_cards"] == [Card.HEART_4]
    assert observation_2["played_cards"] == [Card.HEART_4, Card.HEART_2]
    assert observation_3["played_cards"] == [Card.HEART_4, Card.HEART_2, Card.EXCUSE]

    assert observation_4["plis"][0]["played_cards"] == [Card.HEART_4, Card.HEART_2, Card.EXCUSE, Card.HEART_KING]
    assert environment._current_player == 1
    expected_values = [Card.HEART_KING, Card.HEART_4, Card.HEART_2, Card.EXCUSE]
    assert reward[-1] == get_card_set_point(expected_values)  # last player's team won this round
    assert not done


def test_play_excuse_first():
    environment = setup_environment()[0]
    environment.step(Card.EXCUSE)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    expected_values = [Card.EXCUSE, Card.HEART_KING, Card.HEART_4, Card.HEART_2]
    assert observation["plis"][0]["played_cards"] == expected_values
    assert environment._current_player == 1
    assert reward[-1] == get_card_set_point(expected_values)  # last player's team won this round
    assert not done


def test_play_trump_invalid():
    environment = setup_environment()[0]
    environment._hand_per_player = [[Card.SPADES_1, Card.SPADES_2], [Card.SPADES_3, Card.TRUMP_1],
                                    [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment.step(Card.SPADES_1)
    with pytest.raises(ValueError):
        environment.step(Card.TRUMP_1)


def test_play_trump_below_trump_unallowed():
    environment = setup_environment()[0]
    environment._hand_per_player = [[Card.TRUMP_10, Card.TRUMP_11], [Card.TRUMP_1, Card.TRUMP_12],
                                    [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment.step(Card.TRUMP_10)
    with pytest.raises(ValueError):
        environment.step(Card.TRUMP_1)


def test_pee_not_allowed():
    environment = setup_environment()[0]
    environment._hand_per_player = [[Card.SPADES_1, Card.SPADES_2], [Card.HEART_4, Card.SPADES_3],
                                    [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment.step(Card.SPADES_1)
    with pytest.raises(ValueError):
        environment.step(Card.HEART_4)


def test_play_card_not_in_hand():
    environment = setup_environment()[0]
    environment._hand_per_player = [[Card.SPADES_1], [Card.SPADES_7], [Card.SPADES_3], [Card.SPADES_2]]
    with pytest.raises(ValueError):
        environment.step(Card.HEART_4)


def test_play_two_rounds():
    raise NotImplementedError()


def test_play_complete_game():
    environment = setup_environment()[0]
    raise NotImplementedError()
