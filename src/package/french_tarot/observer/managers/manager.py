from enum import Enum, auto
from typing import Dict, List

from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_manager import AbstractManager
from french_tarot.observer.subscriber import Subscriber


class Manager(AbstractManager):

    def __init__(self):
        super().__init__()
        self._event_subscriber_map: Dict[EventType, List[Subscriber]] = {}

    def add_subscriber(self, subscriber: Subscriber, event_type: 'EventType'):
        if event_type not in self._event_subscriber_map:
            self._event_subscriber_map[event_type] = []
        self._event_subscriber_map[event_type].append(subscriber)

    def publish(self, message: Message):
        if message.event_type in self._event_subscriber_map:
            for subscriber in self._event_subscriber_map[message.event_type]:
                subscriber.push(message.data)


class EventType(Enum):
    DUMMY = auto()
    KILL_ALL = auto()
    OBSERVATION = auto()
    ACTION = auto()
    ACTION_RESULT = auto()
    MODEL_UPDATE = auto()
    RESET_ENVIRONMENT = auto()
