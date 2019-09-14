from torch import nn

from french_tarot.environment.common import CARDS
from french_tarot.play_games import play_trained_agent


def test_play_trained_agent_bid_dog(mocker):
    def mock_create_dog_phase_dqn(_):
        return nn.Sequential(nn.Linear(len(CARDS), 1), nn.Sigmoid())

    mocker.patch('french_tarot.agents.trained_player_bid.BidPhaseAgent._create_dqn', mock_create_dog_phase_dqn)
    mocker.patch('french_tarot.agents.trained_player_bid.BidPhaseAgent.output_dimension', 5)
    mocker.patch('torch.utils.tensorboard.SummaryWriter')
    play_trained_agent.set_all_seeds()
    play_trained_agent._main(n_episodes_training=10)
