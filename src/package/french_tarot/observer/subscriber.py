from abc import abstractmethod
from queue import Queue
from threading import Thread

import dill

from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_manager import AbstractManager
from french_tarot.observer.managers.abstract_subscriber import AbstractSubscriber
from french_tarot.observer.managers.event_type import EventType


class Kill:
    def __init__(self, error: bool = False):
        self.error = error


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
                self._manager.publish(Message(EventType.KILL_ALL, Kill(error=True)))
                output_filepath = "_".join([self.__class__.__name__, str(id(self)), ".dill"])
                print("Dumping state and input at", output_filepath)
                with open(output_filepath, "wb") as f:
                    dill.dump((self, message), f)
                raise e
            finally:
                self._queue.task_done()

    @abstractmethod
    def update(self, data: any):
        pass

    def push(self, data: any):
        self._queue.put(data)
