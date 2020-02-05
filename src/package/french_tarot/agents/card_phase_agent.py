from typing import Dict

import torch
from numpy.random.mtrand import RandomState
from torch import nn, tensor

from french_tarot.agents.agent import Agent, ActionWithProbability
from french_tarot.agents.card_phase_observation_encoder import retrieve_allowed_cards, CardPhaseObservationEncoder
from french_tarot.agents.policy import Policy
from french_tarot.environment.core.core import CARDS
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation


def retrieve_allowed_card_indices(observation):
    return [CARDS.index(card) for card in retrieve_allowed_cards(observation)]


class CardPhaseAgent(Agent):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net
        self._initialize_internals()
        self._random_state = RandomState(seed=1988)
        self._card_phase_observation_encoder = CardPhaseObservationEncoder()

    def update_policy_net(self, policy_net_state_dict: Dict):
        self.policy_net.load_state_dict(policy_net_state_dict)

    def get_action(self, observation: CardPhaseObservation) -> ActionWithProbability:
        self._step += 1
        if not self._random_action_policy.should_play_randomly(self._step):
            self.policy_net.eval()
            with torch.no_grad():
                action_with_probability = self.max_return_action(observation)
        else:
            action_with_probability = self.random_action(observation)
        return action_with_probability

    def max_return_action(self, observation: CardPhaseObservation) -> ActionWithProbability:
        indices = retrieve_allowed_card_indices(observation)
        encode = tensor(self._card_phase_observation_encoder.encode(observation)).float()
        probabilities = torch.softmax(self.policy_net(encode)[indices], dim=0)
        action = self._random_state.choice(indices, p=probabilities.detach().numpy())
        return ActionWithProbability(action=action, probability=probabilities[action])

    def random_action(self, observation: CardPhaseObservation) -> ActionWithProbability:
        indices = retrieve_allowed_card_indices(observation)
        return ActionWithProbability(
            action=self._random_state.choice(indices),
            probability=1. / float(len(indices))
        )

    @property
    def device(self) -> str:
        return "cuda" if next(self.policy_net.parameters()).is_cuda else "cpu"

    def _initialize_internals(self):
        self._step = 0
        self._random_action_policy = Policy()
