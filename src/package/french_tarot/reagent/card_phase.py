from french_tarot.agents.card_phase_observation_encoder import CardPhaseObservationEncoder
from french_tarot.core import create_timestamp
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import CARDS
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation
from french_tarot.reagent.data import ReAgentDataRow


class CardPhaseStateActionEncoder:
    def __init__(self, observation_encoder: CardPhaseObservationEncoder):
        self._current_sequence_number = 0
        self._current_episode_id = 0
        self._observation_encoder = observation_encoder
        self._dataset_id = str(create_timestamp())

    def encode(self, position_towards_taker, observation: CardPhaseObservation, action: Card,
               reward: float) -> ReAgentDataRow:
        possible_actions = self._retrieve_possible_actions(observation)
        self._current_sequence_number += 1
        return ReAgentDataRow(
            mdp_id=self._generate_episode_id(position_towards_taker),
            sequence_number=self._current_sequence_number,
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
