from torch import nn

from environment import Bid, Card
from play_games import play_trained_agent


def test_play_trained_agent(mocker):
    def mock_create_dqn():
        return nn.Sequential(nn.Linear(len(list(Card)), len(list(Bid))))

    mocker.patch('play_games.play_trained_agent.BidPhaseAgent._create_dqn', mock_create_dqn)
    mocker.patch('play_games.play_trained_agent.BidPhaseAgent.output_dimension', 5)
    play_trained_agent._set_all_seeds()
    bid_phase_dqn_agent = play_trained_agent.BidPhaseAgent(
        eps_start=0.5, eps_end=0.05, eps_decay=50, batch_size=4, replay_memory_size=8, device="cpu"
    )
    play_trained_agent._run_training(bid_phase_dqn_agent, n_iterations=10)
