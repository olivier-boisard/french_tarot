import pytest

from environment import Card, FrenchTarotEnvironment


@pytest.fixture(scope="module")
def environment():
    yield FrenchTarotEnvironment()


def test_n_cards():
    assert len(list(Card))


def test_reset_environment(environment):
    observation = environment.reset()
    assert len(observation["hand_per_player"]) == 4
    for hands in observation["hand_per_player"]:
        assert len(hands) == 18
    assert len(observation["dog"]) == 6
    assert "card_played_in_turn" in observation
    assert "card_played_per_player_in_episode" in observation
    assert "who_took" in observation
    assert "is_dog_revealed" in observation
    assert "bid" in observation
    assert "announcements" in observation
