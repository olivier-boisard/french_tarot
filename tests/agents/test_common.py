import pytest

from agents.common import TrainableAgent


def test_create_abstract_agent(mock):
    mock.patch("torch.optim.Adam")
    with pytest.raises(TypeError):
        TrainableAgent(mock.Mock())
