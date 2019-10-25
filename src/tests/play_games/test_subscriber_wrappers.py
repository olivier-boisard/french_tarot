import copy
import datetime
from time import sleep

import pytest
import torch

from french_tarot.agents.common import CoreCardNeuralNet
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgentTrainer, BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgentTrainer, DogPhaseAgent
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer.core import Message
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from french_tarot.observer.subscriber import Subscriber
from french_tarot.play_games.datastructures import ModelUpdate
from french_tarot.play_games.subscriber_wrappers import AllPhaseAgentSubscriber, FrenchTarotEnvironmentSubscriber, \
    ActionResult, TrainerSubscriber, ObservationWithGroup, ActionWithGroup
from src.tests.conftest import create_teardown_func


def test_agent_subscriber(environment: FrenchTarotEnvironment, request):
    manager = Manager()

    agent_subscriber = AllPhaseAgentSubscriber(_create_all_phase_agent(), manager)
    action_subscriber = DummySubscriber(manager)

    request.addfinalizer(create_teardown_func(agent_subscriber, action_subscriber))
    manager.add_subscriber(agent_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(action_subscriber, EventType.ACTION)
    agent_subscriber.start()
    action_subscriber.start()

    observation = environment.reset()
    manager.publish(Message(EventType.OBSERVATION, ObservationWithGroup(0, observation)))
    assert subscriber_receives_data(action_subscriber, ActionWithGroup)


@pytest.mark.timeout(5)
def test_environment_subscriber(request):
    manager = Manager()
    environment_subscriber = FrenchTarotEnvironmentSubscriber(manager)
    observation_subscriber = DummySubscriber(manager)
    action_result_subscriber = DummySubscriber(manager)

    request.addfinalizer(create_teardown_func(environment_subscriber, action_result_subscriber, observation_subscriber))
    manager.add_subscriber(environment_subscriber, EventType.ACTION)
    manager.add_subscriber(observation_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(action_result_subscriber, EventType.ACTION_RESULT)
    environment_subscriber.start()
    action_result_subscriber.start()
    observation_subscriber.start()

    # Test publish data on start
    assert subscriber_receives_data(observation_subscriber, ObservationWithGroup)
    observation = observation_subscriber.data.observation

    # Test publish data after action
    manager.publish(Message(EventType.OBSERVATION, None))
    while observation_subscriber.data is not None:
        pass
    manager.publish(Message(EventType.ACTION, ActionWithGroup(0, RandomPlayer().get_action(observation))))
    assert subscriber_receives_data(observation_subscriber, ObservationWithGroup)

    # Test publish action result after action
    observation = observation_subscriber.data.observation
    manager.publish(Message(EventType.ACTION, RandomPlayer().get_action(observation)))
    assert subscriber_receives_data(action_result_subscriber, ActionResult)


@pytest.mark.timeout(5)
def test_trainer_and_agent_subscribers(environment: FrenchTarotEnvironment, request):
    batch_size = 64
    steps_per_update = 10

    manager = Manager()
    base_card_neural_net = CoreCardNeuralNet()
    bid_phase_agent_model = BidPhaseAgent.create_dqn(base_card_neural_net)
    bid_phase_agent = BidPhaseAgent(bid_phase_agent_model)
    dog_phase_agent_model = DogPhaseAgent.create_dqn(base_card_neural_net)
    dog_phase_agent = DogPhaseAgent(dog_phase_agent_model)
    agent = AllPhaseAgent(bid_phase_agent, dog_phase_agent)
    agent_subscriber = AllPhaseAgentSubscriber(agent, manager)
    bid_phase_trainer = BidPhaseAgentTrainer(bid_phase_agent_model)
    dog_phase_trainer = DogPhaseAgentTrainer(dog_phase_agent_model)
    trainer_subscriber = TrainerSubscriber(bid_phase_trainer, dog_phase_trainer, manager,
                                           steps_per_update=steps_per_update)
    dummy_subscriber = DummySubscriber(manager)
    manager.add_subscriber(trainer_subscriber, EventType.ACTION_RESULT)
    manager.add_subscriber(agent_subscriber, EventType.MODEL_UPDATE)
    manager.add_subscriber(dummy_subscriber, EventType.MODEL_UPDATE)

    untrained_policy_net = copy.deepcopy(bid_phase_agent_model)
    request.addfinalizer(create_teardown_func(trainer_subscriber, dummy_subscriber, agent_subscriber))
    agent_subscriber.start()
    dummy_subscriber.start()
    trainer_subscriber.start()

    observation = environment.reset()
    action = agent.get_action(observation)
    _, reward, done, _ = environment.step(action)

    for _ in range(batch_size):
        manager.publish(Message(EventType.ACTION_RESULT, ActionResult(None, action, observation, reward, done)))

    assert subscriber_receives_data(dummy_subscriber, ModelUpdate)
    untrained_bid_phase_model_weights = _retrieve_parameter_subset(untrained_policy_net)
    trained_bid_phase_model_weights = _retrieve_parameter_subset(bid_phase_agent_model)
    sleep(1)  # TODO ugly
    assert torch.any(trained_bid_phase_model_weights != untrained_bid_phase_model_weights)


class DummySubscriber(Subscriber):

    def __init__(self, message: Message):
        super().__init__(message)
        self.data = None

    def update(self, data: any):
        self.data = data


def _create_all_phase_agent():
    base_card_neural_net = CoreCardNeuralNet()
    bid_phase_agent_model = BidPhaseAgent.create_dqn(base_card_neural_net)
    bid_phase_agent = BidPhaseAgent(bid_phase_agent_model)
    dog_phase_agent_model = DogPhaseAgent.create_dqn(base_card_neural_net)
    dog_phase_agent = DogPhaseAgent(dog_phase_agent_model)
    agent = AllPhaseAgent(bid_phase_agent, dog_phase_agent)
    return agent


def subscriber_receives_data(subscriber, data_type, timeout_seconds=1):
    start_time = datetime.datetime.now()
    received = False
    timeout = False
    while not (received or timeout):
        received = isinstance(subscriber.data, data_type)
        timeout = (datetime.datetime.now() - start_time) >= datetime.timedelta(seconds=timeout_seconds)
    return received


def _retrieve_parameter_subset(model):
    return list(model.parameters())[0]
