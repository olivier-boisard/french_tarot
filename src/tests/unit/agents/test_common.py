import pytest

from french_tarot.agents.neural_net_agent import NeuralNetAgent


def test_create_abstract_agent(mocker):
    mocker.patch("torch.optim.Adam")
    with pytest.raises(TypeError):
        NeuralNetAgent(mocker.Mock())
