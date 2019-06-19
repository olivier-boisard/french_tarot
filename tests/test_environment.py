import pytest

from environment import Card, FrenchTarotEnvironment, GamePhase, Bid


@pytest.fixture(scope="module")
def environment():
    yield FrenchTarotEnvironment()


def test_n_cards():
    assert len(list(Card))


def test_reset_environment(environment):
    observation = environment.reset()
    assert len(observation["hand"]) == 18
    for bid in observation["bid_per_player"]:
        assert bid == Bid.NONE
    assert observation["game_phase"] == GamePhase.BID


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
    with pytest.raises(ValueError):
        environment.step(Bid.PETITE)


def test_announce_poignee(environment):
    assert False
