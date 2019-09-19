from typing import Tuple

import numpy as np
import torch
from numpy.random.mtrand import RandomState
from torch import nn
from torch.nn.modules.loss import BCELoss

from french_tarot.agents.common import BaseNeuralNetAgent, encode_cards, Transition, Trainer
from french_tarot.environment.core import Bid
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation


class BidPhaseAgent(BaseNeuralNetAgent):

    def __init__(self, base_card_neural_net: nn.Module, device: str = "cuda", seed: int = 1988):
        # noinspection PyUnresolvedReferences
        net = BidPhaseAgent._create_dqn(base_card_neural_net).to(device)
        super().__init__(net)
        self._epoch = 0
        self._random_state = RandomState(seed)

    def get_max_return_action(self, observation: BidPhaseObservation):
        state = encode_cards(observation.hand)
        self._step += 1
        output = self.policy_net(state.unsqueeze(0).to(self.device)).argmax().item()
        bid = self._get_bid_value(output, observation.bid_per_player)
        return bid

    def get_random_action(self, observation: BidPhaseObservation):
        output = self._random_state.rand(1, 1)
        bid = self._get_bid_value(output, observation.bid_per_player)
        return bid

    @staticmethod
    def _get_bid_value(estimated_win_probability, bid_per_player):
        if estimated_win_probability >= 0.9:
            bid_value = Bid.GARDE_CONTRE
        elif estimated_win_probability >= 0.8:
            bid_value = Bid.GARDE_SANS
        elif estimated_win_probability >= 0.6:
            bid_value = Bid.GARDE
        elif estimated_win_probability >= 0.5:
            bid_value = Bid.PETITE
        else:
            bid_value = Bid.PASS

        if len(bid_per_player) > 0:
            if np.max(bid_per_player) >= bid_value:
                bid_value = Bid.PASS
        return bid_value

    @property
    def output_dimension(self) -> int:
        return self.policy_net.output_layer[-2].out_features

    @staticmethod
    def _create_dqn(base_neural_net: nn.Module) -> nn.Module:
        width = base_neural_net.output_dimensions
        output_layer = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, 2 * width),
            nn.ReLU(),
            nn.BatchNorm1d(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.ReLU(),

            nn.BatchNorm1d(2 * width),
            nn.Linear(2 * width, 4 * width),
            nn.ReLU(),
            nn.BatchNorm1d(4 * width),
            nn.Linear(4 * width, 4 * width),
            nn.ReLU(),

            nn.BatchNorm1d(4 * width),
            nn.Linear(4 * width, 1),
            nn.Sigmoid()
        )
        return nn.Sequential(base_neural_net, output_layer)


class BidPhaseAgentTrainer(Trainer):

    def _get_input_and_target_tensors(self):
        transitions = self.memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))
        input_vectors = torch.cat(batch.state).to(self.device)
        targets = torch.tensor(batch.reward).float().to(self.device)
        targets[targets >= 0] = 1.
        targets[targets < 0.] = 0
        return input_vectors, targets

    def get_model_output_and_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_vectors, target = self._get_input_and_target_tensors()
        model_output = self._net(input_vectors)
        return model_output, target

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = BCELoss()(model_output.flatten(), target.flatten())
        return loss

    def push_to_memory(self, observation: BidPhaseObservation, action, reward):
        self.memory.push(encode_cards(observation.hand).unsqueeze(0), action, None, reward)
