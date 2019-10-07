import pytest

from french_tarot.observer import Manager, Event, Subscriber, Message


class DummySubscriber(Subscriber):
    def __init__(self):
        super().__init__()
        self.state = None

    def loop_once(self, data: any):
        self.state = data


@pytest.fixture
def dummy_message():
    return Message(event_type=Event.DUMMY, data="Hello!")


@pytest.mark.timeout(3)
def test_lifecycle(dummy_message):
    manager = Manager()
    subscriber = DummySubscriber()
    manager.subscribe(subscriber, Event.DUMMY)

    manager.start()
    subscriber.start()
    assert subscriber._thread.is_alive()

    manager.push(dummy_message)
    _wait_for_subscriber_is_updated(subscriber)

    subscriber.stop()
    manager.stop()

    assert not subscriber._thread.is_alive()
    assert subscriber.state == dummy_message.data
    assert manager._queue.empty()


def _wait_for_subscriber_is_updated(subscriber):
    while not _is_updated(subscriber):
        pass


def _is_updated(subscriber):
    return subscriber.state is not None
