import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agents.common import BaseNeuralNetAgent, BaseCardNeuralNet, encode_card_set
from environment import CARDS

N_FEATURES = 0


def _encode_features(observation: dict) -> torch.Tensor:
    raise NotImplementedError()


class CardPhaseAgent(BaseNeuralNetAgent):

    def __init__(self, base_card_neural_net, device: str = "cuda", **kwargs):
        super(CardPhaseAgent, self).__init__(CardPhaseAgent._create_dqn(base_card_neural_net).to(device), **kwargs)

    def get_action(self, observation: dict):
        hand_vector = encode_card_set(observation["hand"])
        additional_feature_vector = _encode_features(observation)
        output_vector = self._policy_net(torch.cat([hand_vector, additional_feature_vector], dim=1))
        output_vector[~hand_vector] = -np.inf
        return CARDS[output_vector.argmax()]

    def optimize_model(self, tb_writer: SummaryWriter):
        raise NotImplementedError()

    @staticmethod
    def _create_dqn(base_card_neural_net: torch.nn.Module) -> torch.nn.Module:
        return CardPhaseNeuralNet(base_card_neural_net, N_FEATURES)


class CardPhaseNeuralNet(torch.nn.Module):

    def __init__(self, base_card_neural_net: BaseCardNeuralNet, n_additional_features: int):
        super(CardPhaseNeuralNet, self).__init__()
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
