from abc import ABC, abstractmethod

from french_tarot.observer.core import Message


class AbstractSubscriber(ABC):

    @abstractmethod
    def push(self, message: Message):
        raise NotImplementedError
