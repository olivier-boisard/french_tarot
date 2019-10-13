import copy
import datetime

import pytest
import torch
from torch import nn

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgentTrainer
from french_tarot.environment.core import Bid, Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.observer import EventType, Message, Manager, Subscriber
from french_tarot.play_games.datastructures import ModelUpdate
from french_tarot.play_games.subscriber_wrappers import AgentSubscriber, FrenchTarotEnvironmentSubscriber, ActionResult, \
    TrainerSubscriber


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

    # Test publish data on start
    assert subscriber_receives_data(observation_subscriber, Observation)
    observation = observation_subscriber.data

    # Test publish data after action
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
def test_trainer_and_agent_subscribers(request):
    batch_size = 64
    steps_per_update = 10

    manager = Manager()
    agent_subscriber = AgentSubscriber(manager)
    bid_phase_model = agent_subscriber._agent._agents[BidPhaseObservation]._policy_net
    untrained_model = copy.deepcopy(bid_phase_model)
    # noinspection PyUnresolvedReferences
    bid_phase_trainer = BidPhaseAgentTrainer(bid_phase_model)
    dog_phase_trainer = DogPhaseAgentTrainer(nn.Linear(78, 1))
    subscriber = TrainerSubscriber(bid_phase_trainer, dog_phase_trainer, manager, steps_per_update=steps_per_update)
    dummy_subscriber = DummySubscriber()

    untrained_policy_net = _get_policy_net(agent_subscriber)
    agent_subscriber.start()
    dummy_subscriber.start()
    subscriber.start()
    request.addfinalizer(create_teardown_func(subscriber, dummy_subscriber, agent_subscriber))

    manager.add_subscriber(subscriber, EventType.ACTION_RESULT)
    manager.add_subscriber(dummy_subscriber, EventType.MODEL_UPDATE)
    manager.add_subscriber(agent_subscriber, EventType.MODEL_UPDATE)

    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    agent = AllPhaseAgent()
    action = agent.get_action(observation)
    observation, reward, done, _ = environment.step(action)

    for _ in range(batch_size):
        manager.publish(Message(EventType.ACTION_RESULT, ActionResult(action, observation, reward, done)))

    # noinspection PyUnresolvedReferences
    assert subscriber_receives_data(dummy_subscriber, ModelUpdate)
    untrained_bid_phase_policy_net_weights = untrained_policy_net[0].standard_cards_tower[0].weight
    trained_bid_phase_policy_net_weights = _get_policy_net(agent_subscriber)[0].standard_cards_tower[0].weight
    assert torch.any(trained_bid_phase_policy_net_weights != untrained_bid_phase_policy_net_weights)


def _get_policy_net(agent_subscriber):
    return agent_subscriber._agent._agents[BidPhaseObservation]._policy_net
