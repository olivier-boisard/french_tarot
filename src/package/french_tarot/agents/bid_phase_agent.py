import numpy as np
from numpy.random.mtrand import RandomState
from torch import nn

from french_tarot.agents.encoding import encode_cards_as_tensor
from french_tarot.agents.neural_net_agent import NeuralNetAgent
from french_tarot.environment.core import Bid
from french_tarot.environment.subenvironments.bid.bid_phase_observation import BidPhaseObservation


class BidPhaseAgent(NeuralNetAgent):

    def __init__(self, policy_net: nn.Module, seed: int = 1988):
        super().__init__(policy_net)
        self._epoch = 0
        self._random_state = RandomState(seed)

    def max_return_action(self, observation: BidPhaseObservation):
        state = encode_cards_as_tensor(observation.player.hand)
        self._step += 1
        output = self.policy_net(state.unsqueeze(0).to(self.device)).argmax().item()
        return self._get_bid_value(output, observation.bid_per_player)

    def random_action(self, observation: BidPhaseObservation):
        output = self._random_state.rand(1, 1)
        return self._get_bid_value(output, observation.bid_per_player)

    @property
    def output_dimension(self) -> int:
        return self.policy_net.output_layer[-2].out_features

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
