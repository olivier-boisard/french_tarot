from torch import nn

from environment import Card
from play_games import play_trained_agent_bid_dog


def test_play_trained_agent_bid_dog(mocker):
    def mock_create_dog_phase_dqn(_):
        # noinspection PyTypeChecker
        return nn.Sequential(nn.Linear(len(list(Card)), 1), nn.Sigmoid())

    mocker.patch('play_games.play_trained_agent_bid.BidPhaseAgent._create_dqn', mock_create_dog_phase_dqn)
    mocker.patch('play_games.play_trained_agent_bid.BidPhaseAgent.output_dimension', 5)
    play_trained_agent_bid_dog.set_all_seeds()
    play_trained_agent_bid_dog._main(n_episodes_training=10, n_episodes_testing=10)
