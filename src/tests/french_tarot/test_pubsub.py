import pytest
from attr import dataclass

from french_tarot.observer import Manager, Event, Subscriber


class DummyPublisher:
    pass


class DummySubscriber(Subscriber):
    def __init__(self):
        self.state = None

    def update(self, data: any):
        self.state = data


@dataclass
class DummyData:
    value: int


@pytest.fixture
def dummy_data():
    return DummyData(1988)


def test_subscribe(dummy_data):
    subscriber = DummySubscriber()
    manager = Manager()

    manager.subscribe(subscriber, Event.DUMMY)
    assert manager._event_subscriber_map[Event.DUMMY] == [subscriber]

    assert subscriber.state is None
    manager.notify(Event.DUMMY, dummy_data)
    assert subscriber.state == dummy_data
