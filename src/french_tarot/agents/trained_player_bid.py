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

    def __init__(self, base_card_neural_net: nn.Module, device: str = "cuda", **kwargs):
        # noinspection PyUnresolvedReferences
        super(BidPhaseAgent, self).__init__(BidPhaseAgent._create_dqn(base_card_neural_net).to(device), **kwargs)
        self._epoch = 0

    def get_action(self, observation: BidPhaseObservation):
        state = core(observation.hand)

        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(
            -1. * self._steps_done / self._eps_decay)
        self._steps_done += 1
        if self._random_state.rand() > eps_threshold:
            with torch.no_grad():
                self._policy_net.eval()
                output = self._policy_net(state.unsqueeze(0).to(self.device)).argmax().item()
                self._policy_net.train()  # disable eval mode.
        else:
            output = self._random_state.rand()

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

        if len(observation.bid_per_player) > 0:
            if np.max(observation.bid_per_player) >= output:
                output = Bid.PASS
        return Bid(output)

    def optimize_model(self, tb_writer: SummaryWriter):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        if len(self.memory) > self._batch_size:
            transitions = self.memory.sample(self._batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            wins = torch.tensor(batch.reward).float().to(self.device)
            wins[wins >= 0] = 1.
            wins[wins < 0.] = 0

            predicted_win_probability = self._policy_net(state_batch)
            loss = BCELoss()
            loss_output = loss(predicted_win_probability.flatten(), wins.flatten())
            self.loss.append(loss_output.item())

            self._optimizer.zero_grad()
            loss_output.backward()
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.1)
            self._optimizer.step()

            if self._epoch % 1000 == 0:
                tb_writer.add_scalar("Loss/train/Bid", loss_output.item(), self._epoch)
            self._epoch += 1

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
