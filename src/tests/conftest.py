import datetime

import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer.core import Message
from french_tarot.observer.managers.manager import Manager
from french_tarot.observer.subscriber import Subscriber


@pytest.fixture
def manager():
    return Manager()


@pytest.fixture(scope="module")
def environment():
    return FrenchTarotEnvironment()


def create_teardown_func(*threads):
    def teardown():
        for thread in threads:
            thread.stop()

    return teardown


@pytest.fixture
def random_agent():
    return RandomPlayer()


class DummySubscriber(Subscriber):
    def __init__(self, message: Message):
        super().__init__(message)
        self.data = None

    def update(self, data: any):
        self.data = data

    def dump(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, manager: Manager):
        raise NotImplementedError


def subscriber_receives_data(subscriber, data_type, timeout_seconds=1):
    start_time = datetime.datetime.now()
    received = False
    timeout = False
    while not (received or timeout):
        received = isinstance(subscriber.data, data_type)
        timeout = (datetime.datetime.now() - start_time) >= datetime.timedelta(seconds=timeout_seconds)
    return received
