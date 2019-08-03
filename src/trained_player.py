from torch import nn, tensor, argmax

from environment import Card, Bid, GamePhase


def bid_phase_observation_encoder(observation):
    return tensor([card in observation["hand"] for card in list(Card)]).float()


class BidPhaseAgent:

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.BID:
            raise ValueError("Invalid game phase")

        state = bid_phase_observation_encoder(observation)

        nn_width = 128
        model = nn.Sequential(
            nn.Linear(state.shape[0], nn_width),
            nn.ReLU(),
            nn.Linear(nn_width, len(list(Bid)))
        )
        return Bid(argmax(model(state)).item())
