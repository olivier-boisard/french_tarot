import pytest

from french_tarot.observer import EventType, Subscriber, Message
from src.tests.conftest import create_teardown_func


class DummySubscriber(Subscriber):
    def __init__(self):
        super().__init__()
        self.state = None

    def update(self, data: any, *_):
        self.state = data


@pytest.fixture
def dummy_message():
    return Message(event_type=EventType.DUMMY, data="Hello!")


@pytest.mark.timeout(3)
def test_lifecycle(dummy_message, manager, request):
    subscriber = DummySubscriber()
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
