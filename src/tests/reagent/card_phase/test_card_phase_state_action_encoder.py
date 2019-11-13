import re

import numpy as np
import pandas as pd
import pytest

from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder
from src.tests.conftest import setup_environment_for_card_phase


@pytest.fixture
def card_phase_observation():
    return setup_environment_for_card_phase()[1]


@pytest.fixture
def action(card_phase_observation):
    return card_phase_observation.player.hand[0]


@pytest.fixture
def reward(action):
    return setup_environment_for_card_phase()[0].step(action)[2]


@pytest.fixture
def state_encoder(mock):
    state_encoder = mock.Mock()
    state_encoder.encode.return_value = np.arange(4)
    return state_encoder


def test_encoder(state_encoder, card_phase_observation, action, reward):
    timestamp_format = "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}$"

    encoder = CardPhaseStateActionEncoder(state_encoder)
    output = encoder.encode(card_phase_observation, action, reward)
    later_output = encoder.encode(card_phase_observation, action, reward)

    assert output.mdp_id == 0
    assert _timestamp_format_is_valid(timestamp_format, output.sequence_number)
    assert later_output.sequence_number > output.sequence_number
    assert output.state_features == {0: 0, 1: 1, 2: 2, 3: 3}
    assert output.action == 28
    assert output.reward == reward
    assert output.possible_actions == [2, 5, 6, 13, 18, 25, 26, 28, 30, 36, 42, 47, 51, 59, 66, 68, 70, 77]
    assert output.action_probability is None
    assert _timestamp_format_is_valid(timestamp_format, output.ds)
    assert isinstance(output.dictionary, dict)
    assert isinstance(output.dictionary["state_features"], dict)

    rows = [encoder.encode(card_phase_observation, action, reward) for _ in range(10)]
    df = CardPhaseStateActionEncoder.convert_reagent_datarow_list_to_pandas_dataframe(rows)
    assert isinstance(df, pd.DataFrame)


def test_encode_2_episodes(state_encoder, card_phase_observation, action, reward):
    encoder = CardPhaseStateActionEncoder(state_encoder)

    output = encoder.encode(card_phase_observation, action, reward)
    assert output.mdp_id == 0
    encoder.episode_done()
    output_later = encoder.encode(card_phase_observation, action, reward)
    assert output_later.mdp_id == 1


def _timestamp_format_is_valid(timestamp_format, ds):
    return len(re.findall(timestamp_format, ds))
