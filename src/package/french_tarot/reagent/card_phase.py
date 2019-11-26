import time

from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.environment.core import Card, CARDS
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.reagent.data import ReAgentDataRow


class CardPhaseStateActionEncoder:
    def __init__(self, observation_encoder: CardPhaseObservationEncoder):
        self._current_episode_id = 0
        self._observation_encoder = observation_encoder
        self._dataset_id = str(self._create_timestamp())

    def encode(self, position_towards_taker, observation: CardPhaseObservation, action: Card,
               reward: float) -> ReAgentDataRow:
        possible_actions = self._retrieve_possible_actions(observation)
        return ReAgentDataRow(
            mdp_id=self._generate_episode_id(position_towards_taker),
            sequence_number=self._create_timestamp(),
            state_features=self._retrieve_state_features(observation),
            action=CARDS.index(action),
            reward=reward,
            possible_actions=possible_actions,
            action_probability=1. / len(possible_actions),
            ds=self._dataset_id
        )

    def _generate_episode_id(self, position_towards_taker):
        return "_".join([str(self._current_episode_id), str(position_towards_taker)])

    def episode_done(self):
        self._current_episode_id += 1

    def _retrieve_state_features(self, observation):
        return {key: value for key, value in enumerate(self._observation_encoder.encode(observation))}

    @staticmethod
    def _retrieve_possible_actions(observation):
        return sorted(map(lambda card: CARDS.index(card), observation.player.hand))

    @staticmethod
    def _create_timestamp():
        return int(time.time() * 1000000)
