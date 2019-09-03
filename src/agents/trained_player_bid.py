import math

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import BCELoss

from agents.common import BaseNeuralNetAgent, BaseCardNeuralNet, card_set_encoder, Transition
from environment import Bid, GamePhase


class BidPhaseAgent(BaseNeuralNetAgent):

    def __init__(self, base_card_neural_net: nn.Module = None, device: str = "cuda", **kwargs):
        if base_card_neural_net is None:
            base_card_neural_net = BaseCardNeuralNet()
        # noinspection PyUnresolvedReferences
        super(BidPhaseAgent, self).__init__(BidPhaseAgent._create_dqn(base_card_neural_net).to(device), **kwargs)

    def get_action(self, observation: dict):
        if observation["game_phase"] != GamePhase.BID:
            raise ValueError("Invalid game phase")

        state = card_set_encoder(observation["hand"])

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

        if len(observation["bid_per_player"]) > 0:
            if np.max(observation["bid_per_player"]) >= output:
                output = Bid.PASS
        return Bid(output)

    def optimize_model(self):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        display_interval = 100
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

            if len(self.loss) % display_interval == 0:
                print("Loss for bid agent:", np.mean(self.loss[-display_interval:]))

    @property
    def output_dimension(self) -> int:
        return self._policy_net.output_layer[-2].out_features

    @staticmethod
    def _create_dqn(base_neural_net) -> nn.Module:
        output_layer = nn.Sequential(
            nn.BatchNorm1d(base_neural_net.output_dimensions),
            nn.Linear(base_neural_net.output_dimensions, 1),
            nn.Sigmoid()
        )
        return nn.Sequential(base_neural_net, output_layer)
