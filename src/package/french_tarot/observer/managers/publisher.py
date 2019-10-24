from abc import ABC

from french_tarot.observer.core import Message


class Publisher(ABC):
    def publish(self, message: Message):
        raise NotImplementedError
