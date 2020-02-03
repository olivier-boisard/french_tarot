import pytest
from torch.nn.modules.linear import Linear

from french_tarot.agents.card_phase_agent import CardPhaseAgent
from french_tarot.environment.core.card import Card
from french_tarot.environment.core.core import PlayerData
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation


@pytest.fixture
def agent():
    module = Linear(78, 78)
    return CardPhaseAgent(module)


def test_random_action(agent):
    player_data = PlayerData(0, [Card.SPADES_1, Card.CLOVER_1])
    played_cards_in_round = [Card.SPADES_2]
    observation = CardPhaseObservation(player_data, played_cards_in_round)
    action = agent.random_action(observation)
    assert action == 0


def test_max_return_action():
    raise NotImplementedError
