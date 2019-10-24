import pytest

from french_tarot.observer.core import Message
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from french_tarot.observer.subscriber import Subscriber
from src.tests.conftest import create_teardown_func


class DummySubscriber(Subscriber):
    def __init__(self, message: Message):
        super().__init__(message)
        self.state = None

    def update(self, data: any):
        self.state = data


@pytest.fixture
def dummy_message():
    return Message(event_type=EventType.DUMMY, data="Hello!")


@pytest.mark.timeout(3)
def test_lifecycle(dummy_message, manager, request):
    subscriber = DummySubscriber(Manager())
    manager.add_subscriber(subscriber, EventType.DUMMY)

    subscriber.start()
    request.addfinalizer(create_teardown_func(subscriber))

    manager.publish(dummy_message)
    _wait_for_subscriber_is_updated(subscriber)
    assert subscriber.state == dummy_message.data


def _wait_for_subscriber_is_updated(subscriber):
    while not _is_updated(subscriber):
        pass


def _is_updated(subscriber):
    return subscriber.state is not None
