import pytest

from environment import FrenchTarotEnvironment, Bid, CHELEM, GamePhase, Card


def setup_environment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)
    return environment


def test_invalid_action():
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(Card.SPADES_1)


def test_invalid_action_list():
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step([Card.SPADES_1])


def test_invalid_two_poignees():
    environment = setup_environment()
    cards_list = get_card_list()[:10]
    with pytest.raises(ValueError):
        environment.step([cards_list, cards_list])


def get_card_list():
    return [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
            Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_10, Card.TRUMP_11, Card.TRUMP_12, Card.TRUMP_13, Card.TRUMP_14,
            Card.TRUMP_15]


def test_no_announcements():
    environment = setup_environment()
    observation, reward, done, _ = environment.step([])
    assert observation["announcements"][0] == []
    assert reward == 0
    assert not done


def test_complete_announcement_phase():
    environment = setup_environment()
    environment.step([])
    environment.step([])
    environment.step([])
    observation = environment.step([])[0]
    assert observation["game_phase"] == GamePhase.CARD


def test_announce_chelem():
    environment = setup_environment()
    environment.step([])
    observation, reward, done, _ = environment.step([CHELEM])
    assert observation["announcements"][0] == []
    assert observation["announcements"][1] == [CHELEM]
    assert reward == 0
    assert not done


def test_announce_simple_poignee_valid():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    environment.step([])
    environment.step([])
    environment.step([])
    observation, reward, done, _ = environment.step([environment._hand_per_player[-1][-10:]])
    assert isinstance(observation["announcements"][3], list)
    assert reward == 0
    assert not done


def test_announce_simple_poignee_excuse_refused():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([environment._hand_per_player[-1][-10:]])


def test_announce_simple_poignee_excuse_accepted():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment._hand_per_player[3][15] = Card.EXCUSE
    environment._hand_per_player[3][16] = Card.SPADES_1
    environment._hand_per_player[0][0] = Card.TRUMP_15
    environment._original_dog[5] = Card.TRUMP_14

    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    environment.step([])
    environment.step([])
    environment.step([])

    card_list = [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
                 Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_10, Card.TRUMP_11, Card.TRUMP_12, Card.TRUMP_13, Card.EXCUSE,
                 Card.TRUMP_16]
    observation = environment.step([card_list])[0]
    assert isinstance(observation["announcements"][3], list)
    assert observation["announcements"][3]._revealed_cards == card_list


def test_announce_simple_poignee_no_trump():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([environment._hand_per_player[0][-10:]])


def test_announce_simple_poignee_no_such_cards_in_hand():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        card_list = [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
                     Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_21]
        environment.step([card_list])


def test_announce_poignee_invalid():
    environment = setup_environment()

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([get_card_list()[:9]])
    with pytest.raises(ValueError):
        environment.step([get_card_list()[:11]])
    with pytest.raises(ValueError):
        environment.step([get_card_list()[:14]])
    with pytest.raises(ValueError):
        environment.step([get_card_list()[:16]])
