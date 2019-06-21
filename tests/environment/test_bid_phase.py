import pytest

from environment import Card, FrenchTarotEnvironment, GamePhase, Bid


@pytest.fixture(scope="module")
def environment():
    yield FrenchTarotEnvironment()


def test_n_cards():
    assert len(list(Card)) == 78


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
    assert reward == 0


def test_not_done_if_not_all_pass(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PETITE)
    environment.step(Bid.PASS)
    _, _, done, _ = environment.step(Bid.PASS)
    assert not done


def test_wrong_action_in_bid(environment):
    environment.reset()
    with pytest.raises(ValueError):
        environment.step(Card.SPADES_1)


def test_bid_greater_than_garde(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    observation, _, _, _ = environment.step(Bid.GARDE_SANS)
    assert observation["game_phase"] == GamePhase.ANNOUNCEMENTS


def test_five_bids(environment):
    environment.reset()
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    environment.step(Bid.PASS)
    with pytest.raises(ValueError):
        environment.step(Bid.PASS)