import os
from collections import deque

from french_tarot.observer.core import Message
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from src.tests.conftest import DummySubscriber, create_teardown_func


def test_save_history(request):
    output_file_path = os.path.join("manager_history.dill")

    manager = Manager()
    dummy_subscriber = DummySubscriber(manager)
    manager.add_subscriber(dummy_subscriber, EventType.DUMMY)

    dummy_subscriber.start()
    request.addfinalizer(create_teardown_func(dummy_subscriber))
    request.addfinalizer(lambda: _remove_file_if_exists(output_file_path))
    manager.publish(Message(EventType.DUMMY, "a"))
    manager.publish(Message(EventType.DUMMY, "b"))
    manager.publish(Message(EventType.DUMMY, "c"))

    manager.dump_history(output_file_path)
    assert os.path.isfile(output_file_path)
    history = Manager.load_history(output_file_path)
    assert isinstance(history, dict)
    history_queue = list(history.values())[0]
    assert isinstance(history_queue, deque)
    assert history_queue.pop() == "a"
    assert history_queue.pop() == "b"
    assert history_queue.pop() == "c"


def _remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
