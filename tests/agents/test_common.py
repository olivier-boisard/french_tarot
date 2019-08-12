import pytest

from agents.common import Agent


def test_create_abstract_agent(mock):
    mock.patch("torch.optim.Adam")
    with pytest.raises(RuntimeError):
        Agent(mock.Mock())
