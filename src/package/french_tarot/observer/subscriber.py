from abc import abstractmethod
from queue import Queue
from threading import Thread

from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_manager import AbstractManager
from french_tarot.observer.managers.abstract_subscriber import AbstractSubscriber
from french_tarot.observer.managers.event_type import EventType


class Kill:
    pass


class Subscriber(AbstractSubscriber):
    def __init__(self, manager: AbstractManager):
        self._queue = Queue()
        self._thread = Thread(target=self.loop)
        self._manager = manager
        self._manager.add_subscriber(self, EventType.KILL_ALL)

    def start(self):
        self.setup()
        self._thread.start()

    def stop(self):
        self.push(Kill())
        self._thread.join()
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass

    def loop(self):
        run = True
        while run:
            message = self._queue.get()
            try:
                if not isinstance(message, Kill):
                    self.update(message)
                else:
                    run = False
            except Exception as e:
                self._manager.publish(Message(EventType.KILL_ALL, Kill()))
                raise e
            finally:
                self._queue.task_done()

    @abstractmethod
    def update(self, data: any):
        pass

    def push(self, data: any):
        self._queue.put(data)
