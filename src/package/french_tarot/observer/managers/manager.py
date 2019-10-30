import itertools
from typing import Dict, List

import dill

from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_manager import AbstractManager
from french_tarot.observer.managers.abstract_subscriber import AbstractSubscriber
from french_tarot.observer.managers.event_type import EventType


class Manager(AbstractManager):

    def __init__(self):
        super().__init__()
        self._event_subscriber_map: Dict[EventType, List[AbstractSubscriber]] = {}

    def add_subscriber(self, subscriber: AbstractSubscriber, event_type: 'EventType'):
        if event_type not in self._event_subscriber_map:
            self._event_subscriber_map[event_type] = []
        self._event_subscriber_map[event_type].append(subscriber)

    def publish(self, message: Message):
        if message.event_type in self._event_subscriber_map:
            for subscriber in self._event_subscriber_map[message.event_type]:
                subscriber.push(message.data)

    def dump_history(self, output_path: str):
        subscribers = set(itertools.chain(*self._event_subscriber_map.values()))
        output = {subscriber.__class__.__name__: subscriber.input_history for subscriber in subscribers}
        with open(output_path, "wb") as f:
            dill.dump(output, f)

    @classmethod
    def load_history(cls, path):
        with open(path, "rb") as f:
            history = dill.load(f)
        return history
