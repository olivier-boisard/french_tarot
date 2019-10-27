from abc import abstractmethod
from queue import Queue
from threading import Thread

import dill

from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_manager import AbstractManager
from french_tarot.observer.managers.abstract_subscriber import AbstractSubscriber
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager


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
                state_filepath = "_".join([self.__class__.__name__, str(id(self))])
                print("Dumping state at", state_filepath)
                self.dump(state_filepath)
                input_filepath = state_filepath + "_input.dill"
                print("Dumping input at", input_filepath)
                with open(input_filepath) as f:
                    dill.dump(message, f)
                raise e
            finally:
                self._queue.task_done()

    @abstractmethod
    def dump(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, manager: Manager):
        pass

    @abstractmethod
    def update(self, data: any):
        pass

    def push(self, data: any):
        self._queue.put(data)
