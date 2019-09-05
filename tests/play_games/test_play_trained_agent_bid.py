from torch import nn

import agents.common
from agents.common import BaseCardNeuralNet
from agents.trained_player_bid import BidPhaseAgent
from environment import Card
from play_games import play_trained_agent_bid


def test_play_trained_agent_bid(mocker):
    def mock_create_dqn(_):
        # noinspection PyTypeChecker
        return nn.Sequential(nn.Linear(len(list(Card)), 1), nn.Sigmoid())

    mocker.patch('play_games.play_trained_agent_bid.BidPhaseAgent._create_dqn', mock_create_dqn)
    mocker.patch('play_games.play_trained_agent_bid.BidPhaseAgent.output_dimension', 5)
    mocker.patch('torch.utils.tensorboard.SummaryWriter')
    agents.common.set_all_seeds()
    bid_phase_dqn_agent = BidPhaseAgent(
        BaseCardNeuralNet(),
        eps_start=0.5, eps_end=0.05, eps_decay=50, batch_size=4, replay_memory_size=8, device="cpu"
    )
    play_trained_agent_bid._run_training(bid_phase_dqn_agent, n_iterations=10)
