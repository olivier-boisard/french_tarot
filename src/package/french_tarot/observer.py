from abc import abstractmethod, ABC
from enum import Enum, auto
from queue import Queue, Empty
from threading import Thread
from typing import Dict, List

from attr import dataclass


class Subscriber(ABC):
    def __init__(self):
        self._queue = Queue()
        self._running = False
        self._thread = Thread(target=self.loop)

    def start(self):
        self.setup()
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()
        self.teardown()

    def setup(self):
        pass

    def teardown(self):
        pass

    def loop(self):
        while self._running:
            try:
                self.update(self._queue.get_nowait())
            except Empty:
                pass

    def push(self, data: any):
        self._queue.put(data)

    @abstractmethod
    def update(self, data: any):
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
        for subscriber in self._event_subscriber_map[message.event_type]:
            subscriber.push(message.data)


@dataclass
class Message:
    event_type: 'EventType'
    data: any


class EventType(Enum):
    DUMMY = auto()
    OBSERVATION = auto()
    ACTION = auto()
