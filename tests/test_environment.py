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
    observation = environment.reset()
    environment.step(Bid.NONE)


def test_bid_petite(environment):
    observation = environment.reset()
    environment.step(Bid.PETITE)


def test_bid_petite_after_garde(environment):
    observation = environment.reset()
    environment.step(Bid.GARDE)
    with pytest.raises(ValueError):
        assert environment.step(Bid.PETITE)
