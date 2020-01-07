import pytest

from french_tarot.environment.core.bid import Bid
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import CARDS
from french_tarot.environment.french_tarot_environment import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.announcements.announcements_phase_observation import \
    AnnouncementsPhaseObservation
from french_tarot.environment.subenvironments.bid.bid_phase_observation import BidPhaseObservation
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


def test_bid_completed(environment, random_agent):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.GARDE_SANS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step([])
    environment.step([])
    environment.step([])
    observation_for_player_position_0 = environment.step([])[0]
    observation_for_player_position_1 = environment.step(random_agent.get_action(observation_for_player_position_0))[0]
    observation_for_player_position_2 = environment.step(random_agent.get_action(observation_for_player_position_1))[0]
    observation_for_player_position_3 = environment.step(random_agent.get_action(observation_for_player_position_2))[0]

    assert observation_for_player_position_1.player.position_towards_taker == 0
    assert observation_for_player_position_2.player.position_towards_taker == 1
    assert observation_for_player_position_3.player.position_towards_taker == 2
    assert observation_for_player_position_0.player.position_towards_taker == 3


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
    assert isinstance(observation, AnnouncementsPhaseObservation)


def test_five_bids(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    with pytest.raises(FrenchTarotException):
        environment.step(Bid.PASS)
