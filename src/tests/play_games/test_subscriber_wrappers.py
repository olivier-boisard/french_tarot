import pytest

from french_tarot.environment.core import Bid
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer import EventType, Message, Manager, Subscriber
from french_tarot.play_games.subscriber_wrappers import AgentSubscriber


class DummySubscriber(Subscriber):

    def __init__(self):
        super().__init__()
        self.data = None

    def update(self, data: any):
        self.data = data


@pytest.mark.timeout(10)
def test_agent_subscriber(environment: FrenchTarotEnvironment):
    manager = Manager()
    subscriber = AgentSubscriber(manager)
    dummy_subscriber = DummySubscriber()
    manager.add_subscriber(subscriber, EventType.OBSERVATION)
    manager.add_subscriber(dummy_subscriber, EventType.ACTION)

    subscriber.start()
    dummy_subscriber.start()
    observation = environment.reset()
    manager.publish(Message(EventType.OBSERVATION, observation))
    while not isinstance(dummy_subscriber.data, Bid):
        continue
    dummy_subscriber.stop()
    subscriber.stop()
