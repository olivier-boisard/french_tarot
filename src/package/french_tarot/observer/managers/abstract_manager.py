from abc import ABC

from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_subscriber import AbstractSubscriber
from french_tarot.observer.managers.event_type import EventType


class AbstractManager(ABC):

    def add_subscriber(self, subscriber: AbstractSubscriber, event_type: EventType):
        raise NotImplementedError

    def publish(self, message: Message):
        raise NotImplementedError
