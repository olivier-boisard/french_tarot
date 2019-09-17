import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import BCELoss
from torch.utils.tensorboard import SummaryWriter

from french_tarot.agents.common import BaseNeuralNetAgent, core, Transition, OptimizerWrapper
from french_tarot.environment.common import Bid
from french_tarot.environment.observations import BidPhaseObservation


class BidPhaseAgent(BaseNeuralNetAgent):

    def __init__(self, base_card_neural_net: nn.Module, device: str = "cuda", summary_writer: SummaryWriter = None,
                 **kwargs):
        net = BidPhaseAgent._create_dqn(base_card_neural_net).to(device)
        # noinspection PyUnresolvedReferences
        super().__init__(net, BidPhaseAgentOptimizer(net), **kwargs)
        self._epoch = 0
        self._summary_writer = summary_writer

    def get_action_wrapped(self, observation: BidPhaseObservation):
        state = core(observation.hand)

        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(
            -1. * self._step / self._eps_decay)
        self._step += 1
        if self._random_state.rand() > eps_threshold:
            with torch.no_grad():
                self.disable_training()
                output = self._policy_net(state.unsqueeze(0).to(self.device)).argmax().item()
                self.enable_training()
        else:
            output = self._random_state.rand()
        output = self._get_bid_value(output)

        if len(observation.bid_per_player) > 0:
            if np.max(observation.bid_per_player) >= output:
                output = Bid.PASS
        return Bid(output)

    @staticmethod
    def _get_bid_value(estimated_win_probability):
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
        return bid_value

    def get_model_output_and_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_vectors, target = self._get_input_and_target_tensors()
        model_output = self._policy_net(input_vectors)
        return model_output, target

    def _get_input_and_target_tensors(self):
        transitions = self.memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))
        input_vectors = torch.cat(batch.state).to(self.device)
        targets = torch.tensor(batch.reward).float().to(self.device)
        targets[targets >= 0] = 1.
        targets[targets < 0.] = 0
        return input_vectors, targets

    @property
    def output_dimension(self) -> int:
        return self._policy_net.output_layer[-2].out_features

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


class BidPhaseAgentOptimizer(OptimizerWrapper):

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = BCELoss()(model_output.flatten(), target.flatten())
        return loss
