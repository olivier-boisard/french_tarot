from abc import abstractmethod, ABC
from enum import Enum, auto
from queue import Queue, Empty
from threading import Thread
from typing import Dict, List

from attr import dataclass


class Threaded:
    def __init__(self):
        self._queue = Queue()
        self._running = False
        self._thread = Thread(target=self.loop)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()

    def loop(self):
        while self._running:
            try:
                self.loop_once(self._queue.get_nowait())
            except Empty:
                pass

    @abstractmethod
    def loop_once(self, data: any):
        pass

    def push(self, data: any):
        self._queue.put(data)


class Subscriber(Threaded, ABC):
    pass


class Manager(Threaded):

    def __init__(self):
        super().__init__()
        self._event_subscriber_map: Dict[Event, List[Subscriber]] = {}

    def subscribe(self, subscriber: Subscriber, event_type: 'Event'):
        if event_type not in self._event_subscriber_map:
            self._event_subscriber_map[event_type] = []
        self._event_subscriber_map[event_type].append(subscriber)

    def loop_once(self, message: 'Message'):
        for subscriber in self._event_subscriber_map[message.event_type]:
            subscriber.push(message.data)


@dataclass
class Message:
    event_type: 'Event'
    data: any


class Event(Enum):
    DUMMY = auto()
