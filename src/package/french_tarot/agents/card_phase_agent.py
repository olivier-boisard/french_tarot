from typing import Dict

import torch
from numpy.random.mtrand import RandomState
from torch import nn

from french_tarot.agents.agent import Agent, ActionWithProbability
from french_tarot.agents.card_phase_observation_encoder import retrieve_allowed_cards
from french_tarot.agents.policy import Policy
from french_tarot.environment.core.core import CARDS
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation


class CardPhaseAgent(Agent):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net
        self._initialize_internals()
        self._random_state = RandomState(seed=1988)

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
        pass

    def random_action(self, observation: CardPhaseObservation) -> ActionWithProbability:
        allowed_cards = retrieve_allowed_cards(observation)
        return ActionWithProbability(
            action=CARDS.index(self._random_state.choice(allowed_cards)),
            probability=1. / float(len(allowed_cards))
        )

    @property
    def device(self) -> str:
        return "cuda" if next(self.policy_net.parameters()).is_cuda else "cpu"

    def _initialize_internals(self):
        self._step = 0
        self._random_action_policy = Policy()
