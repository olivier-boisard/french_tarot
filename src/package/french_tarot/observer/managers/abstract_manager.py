from abc import ABC

from french_tarot.observer.core import Message
from french_tarot.observer.managers.manager import EventType
from french_tarot.observer.subscriber import Subscriber


class AbstractManager(ABC):
    def add_subscriber(self, subscriber: Subscriber, event_type: EventType):
        raise NotImplementedError

    def publish(self, message: Message):
        raise NotImplementedError
