from abc import ABC, abstractmethod
from typing import Dict

import torch
from french_tarot.agents.training import Policy
from torch import nn

from french_tarot.agents.agent import Agent


class NeuralNetAgent(Agent, ABC):
    def __init__(self, policy_net: nn.Module):
        self.policy_net = policy_net
        self._initialize_internals()

    def update_policy_net(self, policy_net_state_dict: Dict):
        self.policy_net.load_state_dict(policy_net_state_dict)

    def get_action(self, observation):
        self._step += 1
        if not self._random_action_policy.should_play_randomly(self._step):
            self.policy_net.eval()
            with torch.no_grad():
                action = self.max_return_action(observation)
        else:
            action = self.random_action(observation)
        return action

    @abstractmethod
    def max_return_action(self, observation):
        pass

    @abstractmethod
    def random_action(self, observation):
        pass

    @property
    def device(self) -> str:
        return "cuda" if next(self.policy_net.parameters()).is_cuda else "cpu"

    def _initialize_internals(self):
        self._step = 0
        self._random_action_policy = Policy()
