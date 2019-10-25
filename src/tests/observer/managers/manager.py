import os

from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from src.tests.conftest import DummySubscriber, create_teardown_func


def test_save_history(request):
    manager = Manager()
    dummy_subscriber = DummySubscriber(manager)
    manager.add_subscriber(dummy_subscriber, EventType.DUMMY)

    dummy_subscriber.start()
    request.addfinalizer(create_teardown_func(dummy_subscriber))

    output_file_path = os.path.join("manager_history.dill")
    manager.dump_history(output_file_path)
    assert os.path.isfile(output_file_path)
