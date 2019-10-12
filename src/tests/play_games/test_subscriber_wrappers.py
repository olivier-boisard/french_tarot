import datetime
from unittest.mock import Mock

import pytest

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.environment.core import Bid, Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer import EventType, Message, Manager, Subscriber
from french_tarot.play_games.subscriber_wrappers import AgentSubscriber, FrenchTarotEnvironmentSubscriber, ActionResult, \
    TrainerSubscriber, ModelUpdate


class DummySubscriber(Subscriber):

    def __init__(self):
        super().__init__()
        self.data = None

    def update(self, data: any):
        self.data = data


def subscriber_receives_data(subscriber, data_type, timeout_seconds=1):
    start_time = datetime.datetime.now()
    received = False
    timeout = False
    while not (received or timeout):
        received = isinstance(subscriber.data, data_type)
        timeout = (datetime.datetime.now() - start_time) >= datetime.timedelta(seconds=timeout_seconds)
    return received


def create_teardown_func(*threads):
    def teardown():
        for thread in threads:
            thread.stop()

    return teardown


def test_agent_subscriber(environment: FrenchTarotEnvironment, request):
    manager = Manager()
    subscriber = AgentSubscriber(manager)
    dummy_subscriber = DummySubscriber()
    manager.add_subscriber(subscriber, EventType.OBSERVATION)
    manager.add_subscriber(dummy_subscriber, EventType.ACTION)
    request.addfinalizer(create_teardown_func(subscriber, dummy_subscriber))

    subscriber.start()
    dummy_subscriber.start()

    observation = environment.reset()
    manager.publish(Message(EventType.OBSERVATION, observation))
    assert subscriber_receives_data(dummy_subscriber, Bid)


@pytest.mark.timeout(5)
def test_environment_subscriber(environment: FrenchTarotEnvironment, request):
    manager = Manager()

    subscriber = FrenchTarotEnvironmentSubscriber(manager)
    observation_subscriber = DummySubscriber()
    action_result_subscriber = DummySubscriber()
    request.addfinalizer(create_teardown_func(subscriber, action_result_subscriber, observation_subscriber))

    manager.add_subscriber(subscriber, EventType.ACTION)
    manager.add_subscriber(observation_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(action_result_subscriber, EventType.ACTION_RESULT)

    subscriber.start()
    action_result_subscriber.start()
    observation_subscriber.start()

    # Test publish observation on start
    assert subscriber_receives_data(observation_subscriber, Observation)
    observation = observation_subscriber.data

    # Test publish observation after action
    manager.publish(Message(EventType.OBSERVATION, None))
    while observation_subscriber.data is not None:
        pass
    manager.publish(Message(EventType.ACTION, RandomPlayer().get_action(observation)))
    assert subscriber_receives_data(observation_subscriber, Observation)

    # Test publish action result after action
    observation = observation_subscriber.data
    manager.publish(Message(EventType.ACTION, RandomPlayer().get_action(observation)))
    assert subscriber_receives_data(action_result_subscriber, ActionResult)


@pytest.mark.timeout(5)
def test_trainer_subscriber(request):
    batch_size = 64
    steps_per_update = 10

    bid_phase_trainer = Mock()
    dog_phase_trainer = Mock()
    subscriber = TrainerSubscriber(bid_phase_trainer, dog_phase_trainer)
    dummy_subscriber = DummySubscriber()
    dummy_subscriber.start()
    subscriber.start()
    request.addfinalizer(create_teardown_func(subscriber, dummy_subscriber))

    manager = Manager()
    manager.add_subscriber(subscriber, EventType.ACTION_RESULT)
    manager.add_subscriber(dummy_subscriber, EventType.MODEL)

    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    agent = AllPhaseAgent()
    action = agent.get_action(observation)
    observation, reward, done, _ = environment.step(action)

    for _ in range(batch_size):
        manager.publish(Message(EventType.ACTION_RESULT, ActionResult(action, observation, reward, done)))

    # Test behavior
    while bid_phase_trainer.push_to_memory.call_count == 0:
        pass
    bid_phase_trainer.optimize_model.assert_called()
    dog_phase_trainer.optimize_model.assert_called()
    while bid_phase_trainer.push_to_memory.call_count < steps_per_update:
        pass
    assert subscriber_receives_data(dummy_subscriber, ModelUpdate)
