import pytest

from french_tarot.environment.core import Bid, Card, ChelemAnnouncement, CARDS, PoigneeAnnouncement
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.exceptions import FrenchTarotException


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
    with pytest.raises(FrenchTarotException):
        environment.step(Card.SPADES_1)


def test_invalid_action_list():
    environment = setup_environment()
    with pytest.raises(FrenchTarotException):
        environment.step([Card.SPADES_1])


def test_invalid_two_poignees():
    environment = setup_environment()
    cards_list = get_card_list()[:10]
    with pytest.raises(FrenchTarotException):
        environment.step([cards_list, cards_list])


def get_card_list():
    return [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
            Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_10, Card.TRUMP_11, Card.TRUMP_12, Card.TRUMP_13, Card.TRUMP_14,
            Card.TRUMP_15]


def test_no_announcements():
    environment = setup_environment()
    observation, reward, done, _ = environment.step([])
    assert environment._announcements[0] == []
    assert reward == 0
    assert not done


def test_complete_announcement_phase():
    environment = setup_environment()
    environment.step([])
    environment.step([])
    environment.step([])
    observation = environment.step([])[0]
    assert isinstance(observation, CardPhaseObservation)


def test_announce_chelem_by_non_taking_player():
    environment = setup_environment()
    environment.step([])
    with pytest.raises(FrenchTarotException):
        environment.step([ChelemAnnouncement()])


def test_announce_simple_poignee_valid():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)
    poignee = PoigneeAnnouncement.largest_possible_poignee_factory(environment._hand_per_player[0])
    observation, reward, done, _ = environment.step([poignee])
    assert isinstance(environment._announcements[0], list)
    assert reward == 0
    assert not done


def test_announce_chelem_player0():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)

    announcements = [ChelemAnnouncement()]
    observation, reward, done, _ = environment.step(announcements)
    assert isinstance(environment._announcements[0][0], ChelemAnnouncement)
    assert reward == 0
    assert not done


def test_announce_chelem_wrong_string():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)

    with pytest.raises(FrenchTarotException):
        environment.step(["test"])


def test_announce_simple_poignee_excuse_refused():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment._hand_per_player[-1][-1] = Card.EXCUSE
    environment._original_dog[-1] = Card.TRUMP_16
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(FrenchTarotException):
        environment.step([environment._hand_per_player[-1][-10:]])


def test_announce_simple_poignee_excuse_accepted():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment._hand_per_player[3][15] = Card.EXCUSE
    environment._hand_per_player[3][16] = Card.SPADES_1
    environment._hand_per_player[0][0] = Card.TRUMP_15
    environment._original_dog[5] = Card.TRUMP_14

    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    card_list = [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
                 Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_10, Card.TRUMP_11, Card.TRUMP_12, Card.TRUMP_13, Card.TRUMP_16,
                 Card.EXCUSE]
    environment.step([PoigneeAnnouncement.largest_possible_poignee_factory(card_list)])
    assert isinstance(environment._announcements[0][0], PoigneeAnnouncement)
    assert environment._announcements[0][0].revealed_cards == card_list


def test_announce_simple_poignee_no_trump():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(FrenchTarotException):
        environment.step([environment._hand_per_player[0][-10:]])


def test_announce_simple_poignee_no_such_cards_in_hand():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(FrenchTarotException):
        card_list = [Card.TRUMP_1, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7,
                     Card.TRUMP_8, Card.TRUMP_9, Card.TRUMP_21]
        environment.step([card_list])


def test_announce_poignee_invalid():
    environment = setup_environment()

    environment.step([])
    environment.step([])
    environment.step([])
    with pytest.raises(FrenchTarotException):
        environment.step([get_card_list()[:9]])
    with pytest.raises(FrenchTarotException):
        environment.step([get_card_list()[:11]])
    with pytest.raises(FrenchTarotException):
        environment.step([get_card_list()[:14]])
    with pytest.raises(FrenchTarotException):
        environment.step([get_card_list()[:16]])
