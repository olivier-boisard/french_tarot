import math

import numpy as np
import torch
from torch import tensor

from agents.common import Agent, TrainedPlayerNetwork
from environment import Card, Bid, GamePhase


def bid_phase_observation_encoder(observation):
    return tensor([card in observation["hand"] for card in list(Card)]).float()


class BidPhaseAgent(Agent):

    def __init__(self, device="cuda", **kwargs):
        super(BidPhaseAgent, self).__init__(BidPhaseAgent._create_dqn().to(device), **kwargs)

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.BID:
            raise ValueError("Invalid game phase")

        state = bid_phase_observation_encoder(observation)

        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(
            -1. * self._steps_done / self._eps_decay)
        self._steps_done += 1
        if self._random_state.rand() > eps_threshold:
            with torch.no_grad():
                self._policy_net.eval()
                output = self._policy_net(state.unsqueeze(0).to(self.device)).argmax().item()
                self._policy_net.train()  # disable eval mode.
        else:
            output = self._random_state.rand(self.output_dimension).item()

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

    @property
    def output_dimension(self):
        return self._policy_net.output_layer[-2].out_features

    @staticmethod
    def _create_dqn():
        return TrainedPlayerNetwork()
