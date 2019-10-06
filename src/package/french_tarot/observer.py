from abc import abstractmethod, ABC
from enum import Enum, auto
from typing import Dict, List


class Event(Enum):
    DUMMY = auto()


class Subscriber(ABC):

    @abstractmethod
    def update(self, data: any):
        pass


class Manager:
    def __init__(self):
        self._event_subscriber_map: Dict[Event, List[Subscriber]] = {}

    def subscribe(self, subscriber: Subscriber, event_type: Event):
        if event_type not in self._event_subscriber_map:
            self._event_subscriber_map[event_type] = []
        self._event_subscriber_map[event_type].append(subscriber)

    def notify(self, event_type: Event, data: any):
        for subscriber in self._event_subscriber_map[event_type]:
            subscriber.update(data)
