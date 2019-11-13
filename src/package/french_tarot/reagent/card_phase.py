import datetime
from typing import List

import pandas as pd

from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.environment.core import Card, CARDS
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.reagent.data import ReAgentDataRow


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

    def episode_done(self):
        self._current_episode_id += 1

    @staticmethod
    def _timestamp():
        return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def convert_reagent_datarow_list_to_pandas_dataframe(input_list: List[ReAgentDataRow]):
        return pd.DataFrame(map(lambda row: row.dictionary, input_list))
