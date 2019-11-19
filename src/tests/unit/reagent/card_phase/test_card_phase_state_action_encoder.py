import pytest

from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder
from src.tests.conftest import setup_environment_for_card_phase


@pytest.fixture
def action(card_phase_observation):
    return card_phase_observation.player.hand[0]


@pytest.fixture
def reward(action):
    return setup_environment_for_card_phase()[0].step(action)[2]


@pytest.fixture
def state_feature_expected_output():
    return {
        0: 0.,
        1: 0.,
        2: 1.,
        3: 0.,
        4: 0.,
        5: 1.,
        6: 1.,
        7: 0.,
        8: 0.,
        9: 0.,
        10: 0.,
        11: 0.,
        12: 0.,
        13: 1.,
        14: 0.,
        15: 0.,
        16: 0.,
        17: 0.,
        18: 1.,
        19: 0.,
        20: 0.,
        21: 0.,
        22: 0.,
        23: 0.,
        24: 0.,
        25: 1.,
        26: 1.,
        27: 0.,
        28: 1.,
        29: 0.,
        30: 1.,
        31: 0.,
        32: 0.,
        33: 0.,
        34: 0.,
        35: 0.,
        36: 1.,
        37: 0.,
        38: 0.,
        39: 0.,
        40: 0.,
        41: 0.,
        42: 1.,
        43: 0.,
        44: 0.,
        45: 0.,
        46: 0.,
        47: 1.,
        48: 0.,
        49: 0.,
        50: 0.,
        51: 1.,
        52: 0.,
        53: 0.,
        54: 0.,
        55: 0.,
        56: 0.,
        57: 0.,
        58: 0.,
        59: 1.,
        60: 0.,
        61: 0.,
        62: 0.,
        63: 0.,
        64: 0.,
        65: 0.,
        66: 1.,
        67: 0.,
        68: 1.,
        69: 0.,
        70: 1.,
        71: 0.,
        72: 0.,
        73: 0.,
        74: 0.,
        75: 0.,
        76: 0.,
        77: 1.
    }


def test_encoder(card_phase_observation, action, reward, state_feature_expected_output):
    timestamp_format = "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}$"
    player_position_towards_taker = 0

    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    output = encoder.encode(player_position_towards_taker, card_phase_observation, action, reward)
    later_output = encoder.encode(player_position_towards_taker, card_phase_observation, action, reward)

    assert output.mdp_id == "0_0"
    assert later_output.sequence_number > output.sequence_number
    assert output.state_features == state_feature_expected_output
    assert all(map(lambda x: isinstance(x, float), output.state_features.values()))
    assert output.action == 28
    assert output.reward == reward
    assert output.possible_actions == [2, 5, 6, 13, 18, 25, 26, 28, 30, 36, 42, 47, 51, 59, 66, 68, 70, 77]
    assert output.action_probability is None
    assert isinstance(output.dictionary, dict)
    assert isinstance(output.dictionary["state_features"], dict)


def test_encode_2_episodes(card_phase_observation, action, reward):
    player_position_towards_taker = 0
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())

    output_a = encoder.encode(player_position_towards_taker, card_phase_observation, action, reward)
    encoder.episode_done()
    output_b = encoder.encode(player_position_towards_taker, card_phase_observation, action, reward)

    assert output_a.mdp_id != output_b.mdp_id
