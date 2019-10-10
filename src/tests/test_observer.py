import pytest

from french_tarot.observer import Event, Subscriber, Message


class DummySubscriber(Subscriber):
    def __init__(self):
        super().__init__()
        self.state = None

    def update(self, data: any):
        self.state = data


@pytest.fixture
def dummy_message():
    return Message(event_type=Event.DUMMY, data="Hello!")


@pytest.mark.timeout(3)
def test_lifecycle(dummy_message, manager):
    subscriber = DummySubscriber()
    manager.add_subscriber(subscriber, Event.DUMMY)

    subscriber.start()
    assert subscriber._thread.is_alive()

    manager.publish(dummy_message)
    _wait_for_subscriber_is_updated(subscriber)

    subscriber.stop()

    assert not subscriber._thread.is_alive()
    assert subscriber.state == dummy_message.data


def _wait_for_subscriber_is_updated(subscriber):
    while not _is_updated(subscriber):
        pass


def _is_updated(subscriber):
    return subscriber.state is not None
