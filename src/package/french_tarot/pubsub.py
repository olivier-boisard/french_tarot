from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

BUFFER = []


class Topic(Enum):
    DUMMY = "dummy"


class Publisher(ABC):

    @staticmethod
    def publish(message: 'Message', service: 'PubSubService') -> any:
        service.add_message_to_queue(message)


class Subscriber(ABC):

    def __init__(self):
        self._received_messages = []

    def pull(self, message):
        self._received_messages.append(message)


class PubSubService:

    def __init__(self):
        self._subscribers_topic_map: Dict[Topic, List[Subscriber]] = {}
        self._messages: List[Message] = []

    def add_message_to_queue(self, message: any):
        self._messages.append(message)

    def add_subscriber(self, subscriber: Subscriber, topic: Topic):
        if topic not in self._subscribers_topic_map:
            self._subscribers_topic_map[topic] = []
        self._subscribers_topic_map[topic].append(subscriber)

    def broadcast(self):
        for message in self._messages:
            subscribers = self._subscribers_topic_map[message.topic]
            for subscriber in subscribers:
                subscriber.pull(message)
        self._messages = []


@dataclass
class Message:
    topic: Topic
    value: any
