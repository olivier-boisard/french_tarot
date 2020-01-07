import numpy as np
import pytest

from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import CARDS
from french_tarot.environment.french_tarot_environment import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.announcements.announcements_phase_observation import \
    AnnouncementsPhaseObservation
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
    environment.reset(CARDS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation = environment.step(Bid.PETITE)[0]
    return environment, observation


def test_make_dog():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation = environment.step(Bid.PASS)[0]
    taker_hand = observation.player.hand
    dog = taker_hand[2:8]
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward > 0
    assert isinstance(observation, AnnouncementsPhaseObservation)
    assert np.all([card not in observation.player.hand for card in dog])


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
    environment = prepare_environment_sorted_deck()[0]
    dog = [Card.DIAMOND_QUEEN, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6]
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward > 0
    assert isinstance(observation, AnnouncementsPhaseObservation)


def test_make_dog_without_trump():
    environment = FrenchTarotEnvironment()
    environment.reset(CARDS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation = environment.step(Bid.PASS)[0]
    dog = list(observation.player.hand[:6])  # taking player is always player 0
    observation, reward, done, _ = environment.step(dog)
    assert np.isclose(reward, 3.0)


def test_dog_with_card_not_in_players_hand():
    environment, observation = prepare_environment_sorted_deck()
    dog = list(observation.player.hand[:6])
    with pytest.raises(FrenchTarotException):
        environment.step(dog)


def test_dog_has_wrong_number_of_cards():
    environment = FrenchTarotEnvironment()
    environment.reset(CARDS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    observation = environment.step(Bid.PASS)[0]
    dog = list(observation.player.hand[:5])  # taking player is always player 0
    with pytest.raises(FrenchTarotException):
        environment.step(dog)
