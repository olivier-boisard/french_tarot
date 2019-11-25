import pytest

from french_tarot.agents.neural_net import BaseNeuralNetAgent


def test_create_abstract_agent(mocker):
    mocker.patch("torch.optim.Adam")
    with pytest.raises(TypeError):
        BaseNeuralNetAgent(mocker.Mock())
