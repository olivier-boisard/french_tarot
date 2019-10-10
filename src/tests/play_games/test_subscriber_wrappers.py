import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.core import Bid, Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer import EventType, Message, Manager, Subscriber
from french_tarot.play_games.subscriber_wrappers import AgentSubscriber, FrenchTarotEnvironmentSubscriber


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


# @pytest.mark.timeout(3)
def test_environment_subscriber(environment: FrenchTarotEnvironment):
    manager = Manager()
    subscriber = FrenchTarotEnvironmentSubscriber(manager)
    dummy_subscriber = DummySubscriber()
    manager.add_subscriber(subscriber, EventType.ACTION)
    manager.add_subscriber(dummy_subscriber, EventType.OBSERVATION)

    # Test publish observation on start
    subscriber.start()
    dummy_subscriber.start()
    while not isinstance(dummy_subscriber.data, Observation):
        continue
    observation = dummy_subscriber.data

    # Test publish observation after action
    manager.publish(Message(EventType.OBSERVATION, None))
    while dummy_subscriber.data is not None:
        continue
    manager.publish(Message(EventType.ACTION, RandomPlayer().get_action(observation)))
    while not isinstance(dummy_subscriber.data, Observation):
        continue

    # TODO test action result

    dummy_subscriber.stop()
    subscriber.stop()
