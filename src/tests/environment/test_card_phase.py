import pytest

from french_tarot.environment.core import get_card_set_point, CARDS, ChelemAnnouncement, Bid, Card, PoigneeAnnouncement, \
    rotate_list
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.card_phase import CardPhaseEnvironment
from french_tarot.exceptions import FrenchTarotException


def setup_environment(taker=0, sorted_deck=False, chelem=False, poignee=False):
    environment = FrenchTarotEnvironment()
    environment.reset()
    if sorted_deck:
        # noinspection PyProtectedMember
        environment._deal(CARDS)
    good = False
    for i in range(environment.n_players):
        if i == taker:
            environment.step(Bid.GARDE_SANS)
            good = True
        else:
            environment.step(Bid.PASS)
    if not good:
        raise FrenchTarotException("No taking player")

    announcements = []
    if chelem:
        announcements.append(ChelemAnnouncement())
    if poignee:
        # noinspection PyProtectedMember
        card_list = list(environment._hand_per_player[0][-11:-1])
        announcements.append(PoigneeAnnouncement.largest_possible_poignee_factory(card_list))
    environment.step(announcements)
    environment.step([])
    environment.step([])
    observation = environment.step([])[0]
    return environment, observation


def test_start_valid_card():
    environment = setup_environment()[0]
    played_card = environment._hand_per_player[0][0]
    observation, reward, done, _ = environment.step(played_card)
    assert observation.played_cards_in_round == [played_card]
    assert reward is None
    assert not done


def test_start_invalid_action():
    environment = setup_environment()[0]
    with pytest.raises(FrenchTarotException):
        environment.step(ChelemAnnouncement())


def test_start_invalid_card_not_in_hand():
    environment = setup_environment()[0]
    played_card = environment._hand_per_player[1][0]
    with pytest.raises(FrenchTarotException):
        environment.step(played_card)


def test_play_complete_round_valid_last_player_team_wins():
    environment = setup_environment()[0]
    environment.step(Card.HEART_1)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    expected_reward = get_card_set_point([Card.HEART_1, Card.HEART_KING, Card.HEART_4, Card.HEART_2])
    assert reward[0] == 0
    assert reward[1] == expected_reward
    assert reward[2] == expected_reward
    assert reward[3] == expected_reward
    assert not done


def test_play_complete_round_valid_last_player_team_loses():
    environment = setup_environment(taker=1)[0]
    environment.step(Card.HEART_1)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    observation, reward, done, _ = environment.step(Card.HEART_2)

    expected_values = [Card.HEART_KING, Card.HEART_4, Card.HEART_2, Card.HEART_1]
    assert reward[0] == get_card_set_point(expected_values)
    assert reward[1] == 0
    assert reward[2] == 0
    assert reward[3] == 0
    assert not done


def test_play_excuse_in_round():
    environment, observation_0 = setup_environment()
    current_phase_environment = environment._current_phase_environment
    current_phase_environment._hand_per_player = rotate_list(current_phase_environment._hand_per_player, 2)
    observation_1 = environment.step(Card.HEART_4)[0]
    observation_2 = environment.step(Card.HEART_2)[0]
    observation_3 = environment.step(Card.EXCUSE)[0]
    observation_4, reward, done, _ = environment.step(Card.HEART_KING)

    assert observation_0.played_cards_in_round == []
    assert observation_1.played_cards_in_round == [Card.HEART_4]
    assert observation_2.played_cards_in_round == [Card.HEART_4, Card.HEART_2]
    assert observation_3.played_cards_in_round == [Card.HEART_4, Card.HEART_2, Card.EXCUSE]

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
    assert reward[-1] == get_card_set_point(expected_values)  # last player's team won this round
    assert not done


def test_play_trump_invalid():
    hand_per_player = [[Card.SPADES_1, Card.SPADES_2], [Card.SPADES_3, Card.TRUMP_1],
                       [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment = _create_card_environment(hand_per_player)
    environment.step(Card.SPADES_1)
    with pytest.raises(FrenchTarotException):
        environment.step(Card.TRUMP_1)


def test_play_trump_below_trump_unallowed():
    hand_per_player = [[Card.TRUMP_10, Card.TRUMP_11], [Card.TRUMP_1, Card.TRUMP_12],
                       [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment = _create_card_environment(hand_per_player)
    environment.step(Card.TRUMP_10)
    with pytest.raises(FrenchTarotException):
        environment.step(Card.TRUMP_1)


def test_pee_not_allowed():
    hand_per_player = [[Card.SPADES_1, Card.SPADES_2], [Card.HEART_4, Card.SPADES_3],
                       [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment = _create_card_environment(hand_per_player)
    environment.step(Card.SPADES_1)
    with pytest.raises(FrenchTarotException):
        environment.step(Card.HEART_4)


def test_play_card_not_in_hand():
    hand_per_player = [[Card.SPADES_1], [Card.SPADES_7], [Card.SPADES_3], [Card.SPADES_2]]
    environment = _create_card_environment(hand_per_player)
    with pytest.raises(FrenchTarotException):
        environment.step(Card.HEART_4)


def _create_card_environment(hand_per_player):
    return CardPhaseEnvironment(hand_per_player, 0, [], [], [], [])


def test_play_twice_same_card():
    hand_per_player = [[Card.SPADES_10, Card.SPADES_2], [Card.HEART_4, Card.SPADES_3],
                       [Card.SPADES_4, Card.SPADES_5], [Card.SPADES_6, Card.SPADES_7]]
    environment = _create_card_environment(hand_per_player)
    environment.step(Card.SPADES_10)
    environment.step(Card.SPADES_3)
    environment.step(Card.SPADES_4)
    environment.step(Card.SPADES_7)
    with pytest.raises(FrenchTarotException):
        environment.step(Card.SPADES_10)


def test_play_trump_win():
    hand_per_player = [[Card.SPADES_10, Card.SPADES_2], [Card.HEART_4, Card.SPADES_3],
                       [Card.TRUMP_1, Card.HEART_5], [Card.SPADES_6, Card.SPADES_7]]
    environment = _create_card_environment(hand_per_player)
    environment.step(Card.SPADES_10)
    environment.step(Card.SPADES_3)
    environment.step(Card.TRUMP_1)
    environment.step(Card.SPADES_7)
    environment.step(Card.HEART_5)


def test_play_complete_game():
    environment = setup_environment()[0]
    environment.step(Card.HEART_1)
    environment.step(Card.HEART_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.HEART_2)
    environment.step(Card.CLOVER_7)
    environment.step(Card.CLOVER_1)
    environment.step(Card.CLOVER_8)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.HEART_9)
    environment.step(Card.HEART_5)
    environment.step(Card.HEART_7)
    environment.step(Card.TRUMP_2)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.SPADES_9)
    environment.step(Card.SPADES_3)
    environment.step(Card.SPADES_8)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.SPADES_6)
    environment.step(Card.SPADES_2)
    environment.step(Card.SPADES_10)
    environment.step(Card.SPADES_5)
    environment.step(Card.TRUMP_12)
    environment.step(Card.TRUMP_21)
    environment.step(Card.TRUMP_11)
    environment.step(Card.TRUMP_17)
    environment.step(Card.TRUMP_5)
    environment.step(Card.TRUMP_15)
    environment.step(Card.TRUMP_16)
    environment.step(Card.TRUMP_3)
    environment.step(Card.CLOVER_2)
    environment.step(Card.CLOVER_3)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.CLOVER_5)
    environment.step(Card.CLOVER_6)
    environment.step(Card.CLOVER_4)
    environment.step(Card.CLOVER_9)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.TRUMP_10)
    environment.step(Card.TRUMP_14)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.SPADES_4)
    environment.step(Card.SPADES_7)
    environment.step(Card.TRUMP_1)
    environment.step(Card.HEART_JACK)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_19)
    environment.step(Card.HEART_3)
    environment.step(Card.TRUMP_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.TRUMP_18)
    environment.step(Card.TRUMP_9)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.CLOVER_10)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.EXCUSE)
    environment.step(Card.HEART_6)
    environment.step(Card.HEART_8)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_KING)
    environment.step(Card.TRUMP_20)
    environment.step(Card.TRUMP_8)
    observation, reward, done, _ = environment.step(Card.SPADES_1)
    assert done
    assert reward[0] == -540
    assert reward[1] == 180
    assert reward[2] == 180
    assert reward[3] == 180


def test_petit_au_bout_taker():
    environment = setup_environment(taker=3, sorted_deck=True, chelem=True)[0]
    environment.step(Card.TRUMP_2)
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.TRUMP_3)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.TRUMP_1)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -760
    assert reward[1] == -760
    assert reward[2] == -760
    assert reward[3] == 2280


def test_poignee():
    environment = setup_environment(taker=3, sorted_deck=True, poignee=True)[0]
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.TRUMP_2)
    environment.step(Card.TRUMP_3)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_1)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -540
    assert reward[1] == -540
    assert reward[2] == -540
    assert reward[3] == 1620


def test_chelem_unannounced():
    environment = setup_environment(taker=3, sorted_deck=True)[0]
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.TRUMP_2)
    environment.step(Card.TRUMP_3)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_1)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -520
    assert reward[1] == -520
    assert reward[2] == -520
    assert reward[3] == 1560


def test_chelem_announced():
    environment = setup_environment(taker=3, sorted_deck=True, chelem=True)[0]
    environment.step(Card.TRUMP_2)
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.TRUMP_3)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_1)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -720
    assert reward[1] == -720
    assert reward[2] == -720
    assert reward[3] == 2160


def test_chelem_announced_with_excuse():
    environment = setup_environment(taker=3, sorted_deck=True, chelem=True)[0]
    environment._original_dog[-1] = Card.TRUMP_1
    environment._hand_per_player[0][2] = Card.EXCUSE
    environment.step(Card.TRUMP_2)
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.TRUMP_3)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.EXCUSE)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -704
    assert reward[1] == -704
    assert reward[2] == -704
    assert reward[3] == 2112


def test_pee_unallowed():
    environment = setup_environment(taker=0, sorted_deck=True)[0]
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    with pytest.raises(FrenchTarotException):
        environment.step(Card.DIAMOND_QUEEN)


def test_chelem_announced_and_failed():
    environment = setup_environment(taker=3, sorted_deck=True, chelem=True)[0]

    current_phase_environment = environment._current_phase_environment
    tmp = current_phase_environment._hand_per_player[0][16]
    current_phase_environment._hand_per_player[0][16] = current_phase_environment._hand_per_player[1][0]
    current_phase_environment._hand_per_player[1][0] = tmp

    environment.step(Card.TRUMP_2)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.SPADES_1)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_4)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_13)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_1)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.TRUMP_3)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -104
    assert reward[1] == -104
    assert reward[2] == -104
    assert reward[3] == 312


def test_chelem_unannounced_and_achieved_by_other_team():
    environment = setup_environment(taker=0, sorted_deck=True)[0]
    environment.step(Card.SPADES_1)
    environment.step(Card.CLOVER_5)
    environment.step(Card.HEART_9)
    environment.step(Card.TRUMP_1)
    environment.step(Card.TRUMP_2)
    environment.step(Card.SPADES_2)
    environment.step(Card.CLOVER_6)
    environment.step(Card.HEART_10)
    environment.step(Card.TRUMP_3)
    environment.step(Card.SPADES_3)
    environment.step(Card.CLOVER_7)
    environment.step(Card.HEART_JACK)
    environment.step(Card.TRUMP_4)
    environment.step(Card.SPADES_4)
    environment.step(Card.CLOVER_8)
    environment.step(Card.HEART_RIDER)
    environment.step(Card.TRUMP_5)
    environment.step(Card.SPADES_5)
    environment.step(Card.CLOVER_9)
    environment.step(Card.HEART_QUEEN)
    environment.step(Card.TRUMP_6)
    environment.step(Card.SPADES_6)
    environment.step(Card.CLOVER_10)
    environment.step(Card.HEART_KING)
    environment.step(Card.TRUMP_7)
    environment.step(Card.SPADES_7)
    environment.step(Card.CLOVER_JACK)
    environment.step(Card.DIAMOND_1)
    environment.step(Card.DIAMOND_QUEEN)
    environment.step(Card.SPADES_8)
    environment.step(Card.CLOVER_RIDER)
    environment.step(Card.DIAMOND_2)
    environment.step(Card.DIAMOND_KING)
    environment.step(Card.SPADES_9)
    environment.step(Card.CLOVER_QUEEN)
    environment.step(Card.DIAMOND_3)
    environment.step(Card.TRUMP_8)
    environment.step(Card.SPADES_10)
    environment.step(Card.CLOVER_KING)
    environment.step(Card.DIAMOND_4)
    environment.step(Card.TRUMP_9)
    environment.step(Card.SPADES_JACK)
    environment.step(Card.HEART_1)
    environment.step(Card.DIAMOND_5)
    environment.step(Card.TRUMP_10)
    environment.step(Card.SPADES_RIDER)
    environment.step(Card.HEART_2)
    environment.step(Card.DIAMOND_6)
    environment.step(Card.TRUMP_11)
    environment.step(Card.SPADES_QUEEN)
    environment.step(Card.HEART_3)
    environment.step(Card.DIAMOND_7)
    environment.step(Card.TRUMP_12)
    environment.step(Card.SPADES_KING)
    environment.step(Card.HEART_4)
    environment.step(Card.DIAMOND_8)
    environment.step(Card.TRUMP_13)
    environment.step(Card.CLOVER_1)
    environment.step(Card.HEART_5)
    environment.step(Card.DIAMOND_9)
    environment.step(Card.TRUMP_14)
    environment.step(Card.CLOVER_2)
    environment.step(Card.HEART_6)
    environment.step(Card.DIAMOND_10)
    environment.step(Card.TRUMP_15)
    environment.step(Card.CLOVER_3)
    environment.step(Card.HEART_7)
    environment.step(Card.DIAMOND_JACK)
    environment.step(Card.TRUMP_16)
    environment.step(Card.CLOVER_4)
    environment.step(Card.HEART_8)
    reward = environment.step(Card.DIAMOND_RIDER)[1]
    assert reward[0] == -1260
    assert reward[1] == 420
    assert reward[2] == 420
    assert reward[3] == 420
