import math

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import BCELoss
from torch.utils.tensorboard import SummaryWriter

from french_tarot.agents.common import BaseNeuralNetAgent, core, Transition
from french_tarot.environment.common import Bid
from french_tarot.environment.observations import BidPhaseObservation


class BidPhaseAgent(BaseNeuralNetAgent):

    def __init__(self, base_card_neural_net: nn.Module, device: str = "cuda", summary_writer: SummaryWriter = None,
                 **kwargs):
        # noinspection PyUnresolvedReferences
        super(BidPhaseAgent, self).__init__(BidPhaseAgent._create_dqn(base_card_neural_net).to(device), **kwargs)
        self._epoch = 0
        self._summary_writer = summary_writer

    def get_action(self, observation: BidPhaseObservation):
        state = core(observation.hand)

        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(
            -1. * self._steps_done / self._eps_decay)
        self._steps_done += 1
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
    def _get_bid_value(output):
        if output >= 0.9:
            output = Bid.GARDE_CONTRE
        elif output >= 0.8:
            output = Bid.GARDE_SANS
        elif output >= 0.6:
            output = Bid.GARDE
        elif output >= 0.5:
            output = Bid.PETITE
        else:
            output = Bid.PASS
        return output

    def enable_training(self):
        self._policy_net.train()

    def disable_training(self):
        self._policy_net.eval()

    def optimize_model(self):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        if len(self.memory) > self._batch_size:
            input_vectors, target_vectors = self._get_input_and_target_tensors()
            model_output = self._policy_net(input_vectors)
            loss = self._compute_loss(model_output, target_vectors)
            self.loss.append(loss.item())
            self._train_model(loss)

            if self._epoch % 1000 == 0 and self._summary_writer is not None:
                self._summary_writer.add_scalar("Loss/train/Bid", loss.item(), self._epoch)
            self._epoch += 1

    def _train_model(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.1)
        self._optimizer.step()

    @staticmethod
    def _compute_loss(predicted_win_probability, wins):
        loss = BCELoss()
        loss_output = loss(predicted_win_probability.flatten(), wins.flatten())
        return loss_output

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
