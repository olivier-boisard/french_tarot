from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

from french_tarot.agents.common import BaseNeuralNetAgent, CoreCardNeuralNet, core, OptimizerWrapper
from french_tarot.environment.common import CARDS
from french_tarot.environment.observations import Observation

FEATURE_VECTOR_SIZE = 16


def _extract_features(observation: dict) -> dict:
    raise NotImplementedError()


def _encode_features(observation: dict) -> torch.Tensor:
    raise NotImplementedError()


class CardPhaseOptimizer(OptimizerWrapper):
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return smooth_l1_loss(model_output, target)


class CardPhaseAgent(BaseNeuralNetAgent):

    def get_model_output_and_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def __init__(self, base_card_neural_net, device: str = "cuda", **kwargs):
        net = CardPhaseAgent._create_dqn(base_card_neural_net).to(device)
        super().__init__(net, CardPhaseOptimizer(net), **kwargs)

    def get_action(self, observation: Observation):
        hand_vector = core(observation.hand)
        additional_feature_vector = _encode_features(_extract_features(observation))
        output_vector = self._policy_net(torch.cat([hand_vector, additional_feature_vector], dim=1))
        output_vector[~hand_vector] = -np.inf
        return CARDS[output_vector.argmax()]

    @staticmethod
    def _create_dqn(base_card_neural_net: torch.nn.Module) -> torch.nn.Module:
        return CardPhaseNeuralNet(base_card_neural_net, FEATURE_VECTOR_SIZE)


class CardPhaseNeuralNet(torch.nn.Module):

    def __init__(self, base_card_neural_net: CoreCardNeuralNet, n_additional_features: int):
        super().__init__()
        self._base_card_neural_net = base_card_neural_net
        n_inputs = base_card_neural_net.output_dimensions + n_additional_features
        nn_width = 256
        self._merge_tower = nn.Sequential(
            nn.BatchNorm1d(n_inputs),
            nn.Linear(n_inputs, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, 2 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(2 * nn_width),
            nn.Linear(2 * nn_width, 2 * nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(2 * nn_width),
            nn.Linear(2 * nn_width, 4 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(4 * nn_width),
            nn.Linear(4 * nn_width, 8 * nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, len(CARDS))
        )

    def forward(self, xx: torch.Tensor) -> torch.Tensor:
        n_cards = len(CARDS)
        xx_base = self._base_card_neural_net(xx[:, :n_cards])
        return self._merge_tower(torch.cat([xx_base, xx[:, n_cards:]]))
