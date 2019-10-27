import os

from french_tarot.observer.managers.manager import Manager
from french_tarot.play_games.subscriber_wrappers import FrenchTarotEnvironmentSubscriber


def test_01():
    path = os.path.join(os.path.dirname(__file__), os.pardir, "resources",
                        "FrenchTarotEnvironmentSubscriber_139937673632624.dill")
    subscriber = FrenchTarotEnvironmentSubscriber.load(path, Manager())
    assert False
