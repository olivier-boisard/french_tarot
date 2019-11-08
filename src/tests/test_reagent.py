import datetime
import re
from _ast import Dict
from dataclasses import dataclass

import numpy as np

from french_tarot.environment.core import Card
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from src.tests.conftest import setup_environment


@dataclass
class ReAgentDataRow:
    mdp_id: int
    sequence_number: str
    state_feature: Dict


class CardPhaseObservationEncoder:
    def encode(self, observation: CardPhaseObservation):
        pass


class CardPhaseStateActionEncoder:
    def __init__(self, observation_encoder: CardPhaseObservationEncoder):
        self._episode_id = 0
        self._observation_encoder = observation_encoder

    def encode(self, observation: CardPhaseObservation, action: Card, reward: float):
        return ReAgentDataRow(
            mdp_id=0,
            sequence_number=datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S.%f"),
            state_feature=self._observation_encoder.encode(observation)
        )


def test_encoder(mock):
    timestamp_format = "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}$"

    state_encoder = mock.Mock()
    state_encoder.encode.return_value = np.arange(4)
    encoder = CardPhaseStateActionEncoder(state_encoder)

    environment, observation = setup_environment()
    action = observation.player.hand[0]
    reward = environment.step(action)[2]

    output = encoder.encode(observation, action, reward)
    later_output = encoder.encode(observation, action, reward)

    assert output.mdp_id == 0
    assert len(re.findall(timestamp_format, output.sequence_number))
    assert later_output.sequence_number > output.sequence_number
    assert output.state_feature == {0: 0, 1: 1, 2: 2, 3: 3}


def test_encode_2_episode():
    raise NotImplementedError
