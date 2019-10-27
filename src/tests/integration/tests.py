import os

import dill

from french_tarot.observer.managers.manager import Manager
from french_tarot.play_games.subscriber_wrappers import FrenchTarotEnvironmentSubscriber
from src.tests.conftest import create_teardown_func


class FrenchTarotEnvironmentSubscriberNoSetup(FrenchTarotEnvironmentSubscriber):

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

    def setup(self):
        pass

    def dump(self):
        pass


def test_01(request):
    state_path = os.path.join(os.path.dirname(__file__), os.pardir, "resources",
                              "FrenchTarotEnvironmentSubscriber_140317080087856")
    subscriber = FrenchTarotEnvironmentSubscriberNoSetup.load(state_path, Manager())
    input_path = state_path + "_input.dill"
    with open(input_path, "rb") as f:
        subscriber_input = dill.load(f)
    subscriber.start()
    request.addfinalizer(create_teardown_func(subscriber))

    subscriber.push(subscriber_input)
    assert False
