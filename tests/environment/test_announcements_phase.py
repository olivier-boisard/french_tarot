import pytest

from environment import FrenchTarotEnvironment, Bid, CHELEM, Poignee, GamePhase, Card


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
        environment.step([Poignee(cards_list)])


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
    observation, reward, done, _ = environment.step([Poignee(environment._hand_per_player[-1][-10:])])
    assert isinstance(observation["announcements"][3], Poignee)
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
        environment.step([Poignee(environment._hand_per_player[-1][-10:])])


def test_announce_simple_poignee_excuse_accepted():
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
    observation = environment.step([Poignee(environment._hand_per_player[-1][-10:])])[0]
    assert isinstance(observation["announcements"][3], Poignee)


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
        environment.step([Poignee(environment._hand_per_player[-1][-10:])])


def test_announce_simple_poignee_no_such_cards_in_hand():
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
        environment.step([Poignee(
            [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
             Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_10])])


def test_announce_poignee_invalid():
    environment = setup_environment()

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([Poignee(get_card_list()[:9])])