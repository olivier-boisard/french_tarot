import numpy as np
import pytest

from french_tarot.environment.common import Card, Bid, CARDS
from french_tarot.environment.environment import FrenchTarotEnvironment
from french_tarot.environment.observations import AnnouncementPhaseObservation
from french_tarot.exceptions import FrenchTarotException


def setup_environment():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    return environment


def prepare_environment_sorted_deck():
    environment = FrenchTarotEnvironment()
    environment.reset()
    # noinspection PyProtectedMember
    environment._deal(CARDS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    return environment


def test_make_dog():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    dog = list(environment._hand_per_player[0][2:8])  # player 0 is always the taking player
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward > 0
    assert isinstance(observation, AnnouncementPhaseObservation)
    taking_players_hand = environment._hand_per_player[0]
    assert np.all([card not in taking_players_hand for card in dog])
    assert len(taking_players_hand) == environment._n_cards_per_player


def test_make_dog_with_duplicated_card():
    dog = [Card.SPADES_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    with pytest.raises(FrenchTarotException):
        environment.step(dog)


def test_make_dog_with_king():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_KING]
    environment = setup_environment()
    with pytest.raises(FrenchTarotException):
        environment.step(dog)


def test_make_dog_with_oudler():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.EXCUSE]
    environment = setup_environment()
    with pytest.raises(FrenchTarotException):
        environment.step(dog)


def test_make_dog_with_trump_invalid():
    dog = [Card.TRUMP_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    with pytest.raises(FrenchTarotException):
        environment.step(dog)


def test_make_dog_with_trump_valid():
    environment = prepare_environment_sorted_deck()
    dog = [Card.DIAMOND_QUEEN, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6]
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward > 0
    assert isinstance(observation, AnnouncementPhaseObservation)


def test_make_dog_without_trump():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    dog = list(environment._hand_per_player[0][:6])  # taking player is always player 0
    observation, reward, done, _ = environment.step(dog)
    assert np.isclose(reward, 3.0)


def test_dog_with_card_not_in_players_hand():
    environment = prepare_environment_sorted_deck()
    dog = list(environment._hand_per_player[0][:6])
    with pytest.raises(FrenchTarotException):
        environment.step(dog)


def test_dog_has_wrong_number_of_cards():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(CARDS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    dog = list(environment._hand_per_player[0][:5])  # taking player is always player 0
    with pytest.raises(FrenchTarotException):
        environment.step(dog)
