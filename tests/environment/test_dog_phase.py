import numpy as np
import pytest

from environment import FrenchTarotEnvironment, Bid, Card, GamePhase


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
    environment._deal(list(Card))
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
    dog = list(environment._hand_per_player[environment._taking_player][2:8])
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward > 0
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS
    taking_players_hand = environment._hand_per_player[environment._taking_player]
    assert np.all([card not in taking_players_hand for card in dog])
    assert len(taking_players_hand) == environment._n_cards_per_player


def test_make_dog_with_duplicated_card():
    dog = [Card.SPADES_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_king():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_KING]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_oudler():
    dog = [Card.SPADES_1, Card.SPADES_2, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.EXCUSE]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_trump_invalid():
    dog = [Card.TRUMP_1, Card.SPADES_1, Card.SPADES_3, Card.SPADES_4, Card.SPADES_5, Card.SPADES_6]
    environment = setup_environment()
    with pytest.raises(ValueError):
        environment.step(dog)


def test_make_dog_with_trump_valid():
    environment = prepare_environment_sorted_deck()
    dog = [Card.DIAMOND_QUEEN, Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6]
    observation, reward, done, _ = environment.step(dog)
    assert not done
    assert reward > 0
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS
    assert observation["revealed_cards_in_dog"] == [Card.TRUMP_2, Card.TRUMP_3, Card.TRUMP_4, Card.TRUMP_5,
                                                    Card.TRUMP_6]


def test_make_dog_without_trump():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    dog = list(environment._hand_per_player[environment._taking_player][:6])
    observation, reward, done, _ = environment.step(dog)
    assert len(observation["revealed_cards_in_dog"]) == 0


def test_dog_with_card_not_in_players_hand():
    environment = prepare_environment_sorted_deck()
    dog = list(environment._hand_per_player[0][:6])
    with pytest.raises(ValueError):
        environment.step(dog)


def test_dog_has_wrong_number_of_cards():
    environment = FrenchTarotEnvironment()
    environment.reset()
    environment._deal(list(Card))
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    dog = list(environment._hand_per_player[environment._taking_player][:5])
    with pytest.raises(ValueError):
        environment.step(dog)


test_make_dog_with_trump_valid()
