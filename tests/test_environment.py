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
    assert len(observation["card_played_in_turn"]) == 0
    assert len(observation["card_played_per_player_in_episode"]) == 0
    assert observation["who_took"] is None
    assert not observation["is_dog_revealed"]
    assert observation["bid"] is None
    assert len(observation["announcements"]) == 0
    assert not observation["is_game_started"]
