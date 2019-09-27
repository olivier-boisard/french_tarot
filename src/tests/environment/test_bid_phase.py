import copy

import numpy as np
import pytest

from french_tarot.environment.core import CARDS, Bid, Card
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.exceptions import FrenchTarotException


@pytest.fixture(scope="module")
def environment():
    yield FrenchTarotEnvironment()


def test_n_cards():
    assert len(CARDS) == 78


def test_reset_environment(environment):
    observation = environment.reset()
    assert len(observation.player.hand) == 18
    for bid in observation.bid_per_player:
        assert bid == Bid.NONE
    assert isinstance(observation, BidPhaseObservation)


def test_bid_pass(environment):
    environment.reset()
    environment.step(Bid.PASS)


def test_bid_petite(environment):
    environment.reset()
    environment.step(Bid.PETITE)


def test_bid_after_petite(environment):
    environment.reset()
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)


def test_bid_petite_after_pass(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)


def test_bid_petite_after_garde(environment):
    environment.reset()
    environment.step(Bid.GARDE)
    with pytest.raises(FrenchTarotException):
        environment.step(Bid.PETITE)


def test_done_if_all_pass(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    _, _, done, _ = environment.step(Bid.PASS)
    assert done


def test_reward_zero(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    _, reward, _, _ = environment.step(Bid.PASS)
    assert reward == [0, 0, 0, 0]


def test_bid_completed(environment):
    environment.reset()
    original_hands = copy.deepcopy(environment._hand_per_player)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    observation, _, done, _ = environment.step(Bid.PASS)
    assert np.all(environment._hand_per_player[0] == original_hands[1])
    assert np.all(environment._hand_per_player[1] == original_hands[2])
    assert np.all(environment._hand_per_player[2] == original_hands[3])
    assert np.all(environment._hand_per_player[3] == original_hands[0])


def test_wrong_action_in_bid(environment):
    environment.reset()
    with pytest.raises(FrenchTarotException):
        environment.step(Card.SPADES_1)


def test_bid_greater_than_garde(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation, _, _, _ = environment.step(Bid.GARDE_SANS)
    assert isinstance(observation, AnnouncementPhaseObservation)


def test_five_bids(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    with pytest.raises(FrenchTarotException):
        environment.step(Bid.PASS)
