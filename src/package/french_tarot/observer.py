from abc import abstractmethod, ABC
from enum import Enum, auto
from queue import Queue
from threading import Thread
from typing import Dict, List, Union

from attr import dataclass


class Kill:
    pass


class Subscriber(ABC):
    def __init__(self):
        self._queue = Queue()
        self._thread = Thread(target=self.loop)

    def start(self):
        self.setup()
        self._thread.start()

    def stop(self):
        self.push(Kill(), None)
        self._thread.join()
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass

    def loop(self):
        run = True
        while run:
            message, group = self._queue.get()
            if not isinstance(message, Kill):
                self.update(message, group)
            else:
                run = False
            self._queue.task_done()

    def push(self, data: any, group: int):
        self._queue.put((data, group))

    @abstractmethod
    def update(self, data: any, group_id: int):
        pass


class Manager:

    def __init__(self):
        super().__init__()
        self._event_subscriber_map: Dict[EventType, List[Subscriber]] = {}

    def add_subscriber(self, subscriber: Subscriber, event_type: 'EventType'):
        if event_type not in self._event_subscriber_map:
            self._event_subscriber_map[event_type] = []
        self._event_subscriber_map[event_type].append(subscriber)

    def publish(self, message: 'Message'):
        if message.event_type in self._event_subscriber_map:
            for subscriber in self._event_subscriber_map[message.event_type]:
                subscriber.push(message.data, message.group)


@dataclass
class Message:
    event_type: 'EventType'
    data: any
    group: Union[int, None] = None


class EventType(Enum):
    DUMMY = auto()
    OBSERVATION = auto()
    ACTION = auto()
    ACTION_RESULT = auto()
    MODEL_UPDATE = auto()
    RESET_ENVIRONMENT = auto()
