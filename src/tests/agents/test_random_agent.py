import numpy as np
import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.core import Bid, Card, PlayerData
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation


def test_instantiate_random_player():
    random_agent = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    action = random_agent.get_action(observation)
    assert isinstance(action, Bid)


def test_randomness_when_bidding():
    random_agent = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    actions = [random_agent.get_action(observation) for _ in range(10)]
    assert len(np.unique(actions)) > 1


@pytest.mark.repeat(10)
def test_play_game(random_agent, environment):
    observation = environment.reset()
    done = False
    cnt = 0
    while not done:
        observation, _, done, _ = environment.step(random_agent.get_action(observation))
        cnt += 1
        if cnt >= 1000:
            raise RuntimeError("Infinite loop")

    # No assert needed here, the code just needs to run without raising exceptions


def test_bugfix_announcement_phase():
    agent = RandomPlayer()
    hand = [Card.TRUMP_14, Card.TRUMP_8, Card.EXCUSE, Card.TRUMP_21, Card.HEART_8, Card.TRUMP_2, Card.CLOVER_KING,
            Card.SPADES_10, Card.DIAMOND_6, Card.SPADES_9, Card.TRUMP_15, Card.SPADES_QUEEN, Card.TRUMP_20,
            Card.SPADES_3, Card.TRUMP_7, Card.TRUMP_10, Card.TRUMP_6, Card.TRUMP_4]
    action = agent.get_action(AnnouncementPhaseObservation(PlayerData(0, hand)))
    assert Card.EXCUSE not in action[0].revealed_cards
