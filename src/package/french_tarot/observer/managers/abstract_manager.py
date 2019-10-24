from abc import ABC

from french_tarot.observer.core import Message


class AbstractManager(ABC):
    def publish(self, message: Message):
        raise NotImplementedError
