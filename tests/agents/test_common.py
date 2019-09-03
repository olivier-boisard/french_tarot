import pytest

from agents.common import BaseNeuralNetAgent


def test_create_abstract_agent(mock):
    mock.patch("torch.optim.Adam")
    with pytest.raises(TypeError):
        BaseNeuralNetAgent(mock.Mock())
