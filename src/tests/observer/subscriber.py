from french_tarot.observer.core import Message
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from src.tests.conftest import DummySubscriber, subscriber_receives_data, create_teardown_func


def test_subscriber_history(request):
    manager = Manager()
    dummy_subscriber = DummySubscriber(manager)
    manager.add_subscriber(dummy_subscriber, EventType.DUMMY)
    manager.publish(Message(EventType.DUMMY, "dummy"))

    dummy_subscriber.start()
    request.addfinalizer(create_teardown_func(dummy_subscriber))

    assert subscriber_receives_data(dummy_subscriber, str)
    assert dummy_subscriber.input_history.pop() == "dummy"
