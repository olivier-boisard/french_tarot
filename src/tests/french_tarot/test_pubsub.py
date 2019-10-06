import pytest

from french_tarot.observer import Manager, Event, Subscriber, Publisher, Message


class DummySubscriber(Subscriber):
    def __init__(self):
        self.state = None

    def update(self, data: any):
        self.state = data


@pytest.fixture
def dummy_message():
    return Message(event_type=Event.DUMMY, data="Hello!")


def test_synchronous_pubsub(dummy_message):
    subscriber = DummySubscriber()
    manager = Manager()

    manager.subscribe(subscriber, Event.DUMMY)
    assert manager._event_subscriber_map[Event.DUMMY] == [subscriber]

    assert subscriber.state is None
    manager.notify(dummy_message)
    assert subscriber.state == dummy_message.data


@pytest.mark.timeout(1)
def test_threaded_pubsub(dummy_message):
    manager = Manager()
    subscriber = DummySubscriber()
    publisher = Publisher(manager)
    manager.subscribe(subscriber, Event.DUMMY)

    manager.start()
    subscriber.start()
    publisher.start()

    publisher.stop()
    subscriber.stop()
    manager.stop()

    assert subscriber.state == dummy_message.data
    assert manager._queue.empty()
