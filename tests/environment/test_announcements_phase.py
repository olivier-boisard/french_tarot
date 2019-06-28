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
    with pytest.raises(ValueError):
        environment.step([Poignee.SIMPLE, Poignee.DOUBLE])


def test_no_announcements():
    environment = setup_environment()
    observation, reward, done, _ = environment.step([])
    assert observation["announcements"][0] == []
    assert reward == 0
    assert not done


def test_announce_chelem():
    environment = setup_environment()
    environment.step([])
    observation, reward, done, _ = environment.step([CHELEM])
    assert observation["announcements"][0] == []
    assert observation["announcements"][1] == [CHELEM]
    assert observation["game_phase"] == GamePhase.CARD
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
    observation, reward, done, _ = environment.step([Poignee.SIMPLE])
    assert observation["announcements"][3] == [Poignee.SIMPLE]
    assert reward == 0
    assert not done


def test_announce_simple_poignee_invalid():
    environment = setup_environment()

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([Poignee.SIMPLE])


def test_announce_double_poignee_invalid():
    environment = setup_environment()

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([Poignee.DOUBLE])


def test_announce_triple_poignee_invalid():
    environment = setup_environment()

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(ValueError):
        environment.step([Poignee.TRIPLE])
