import datetime
import re
from _ast import Dict
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
import pytest

from french_tarot.environment.core import Card, CARDS
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from src.tests.conftest import setup_environment_for_card_phase


@dataclass
class ReAgentDataRow:
    mdp_id: int
    sequence_number: str
    state_features: Dict
    action: int
    reward: float
    possible_actions: List[int]
    action_probability: Union[int, None]
    ds: str

    @property
    def dictionary(self):
        self_as_dict = vars(self)
        self_as_dict["mdp_id"] = str(self_as_dict["mdp_id"])
        self_as_dict["state_features"] = {str(key): str(value) for key, value in self_as_dict["state_features"].items()}
        self_as_dict["action"] = self_as_dict["action"]
        self_as_dict["possible_actions"] = list(map(str, self_as_dict["possible_actions"]))
        return self_as_dict


class CardPhaseObservationEncoder:
    def encode(self, observation: CardPhaseObservation):
        pass


class CardPhaseStateActionEncoder:
    def __init__(self, observation_encoder: CardPhaseObservationEncoder):
        self._current_episode_id = 0
        self._observation_encoder = observation_encoder
        self._dataset_id = self._timestamp()

    def encode(self, observation: CardPhaseObservation, action: Card, reward: float):
        return ReAgentDataRow(
            mdp_id=self._current_episode_id,
            sequence_number=self._timestamp(),
            state_features={key: value for key, value in enumerate(self._observation_encoder.encode(observation))},
            action=CARDS.index(action),
            reward=reward,
            possible_actions=sorted(map(lambda card: CARDS.index(card), observation.player.hand)),
            action_probability=None,
            ds=self._timestamp()
        )

    @staticmethod
    def _timestamp():
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def convert_reagent_datarow_list_to_pandas_dataframe(input_list: List[ReAgentDataRow]):
        return pd.DataFrame(map(lambda row: row.dictionary, input_list))

    def episode_done(self):
        self._current_episode_id += 1


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
