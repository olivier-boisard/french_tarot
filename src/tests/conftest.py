import os

import pytest

from french_tarot.agents.random_agent import RandomAgent
from french_tarot.environment.core.announcements.chelem_announcement import ChelemAnnouncement
from french_tarot.environment.core.announcements.poignee.poignee_announcement import PoigneeAnnouncement
from french_tarot.environment.core.bid import Bid
from french_tarot.environment.french_tarot_environment import FrenchTarotEnvironment
from french_tarot.exceptions import FrenchTarotException


@pytest.fixture(scope="module")
def environment():
    return FrenchTarotEnvironment()


@pytest.fixture
def random_agent():
    return RandomAgent()


# TODO replace by fixture
def setup_environment_for_card_phase(taker=0, shuffled_deck=None, chelem=False, poignee=False):
    environment = FrenchTarotEnvironment()

    observation = environment.reset(shuffled_card_deck=shuffled_deck)
    good = False
    for i in range(environment.n_players):
        if i == taker:
            observation = environment.step(Bid.GARDE_SANS)[0]
            good = True
        else:
            observation = environment.step(Bid.PASS)[0]
    if not good:
        raise FrenchTarotException("No taking player")

    announcements = []
    if chelem:
        announcements.append(ChelemAnnouncement())
    if poignee:
        announcements.append(PoigneeAnnouncement.largest_possible_poignee_factory(observation.player.hand[-11:-1]))
    environment.step(announcements)
    environment.step([])
    environment.step([])
    observation = environment.step([])[0]
    return environment, observation


@pytest.fixture
def resources_folder():
    return os.path.join(os.path.dirname(__file__), os.pardir, 'resources')


@pytest.fixture
def card_phase_observation():
    return setup_environment_for_card_phase()[1]
