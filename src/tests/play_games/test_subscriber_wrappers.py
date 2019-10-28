import pytest

from french_tarot.agents.common import CoreCardNeuralNet
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer.core import Message
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from french_tarot.play_games.subscriber_wrappers import AllPhaseAgentSubscriber, FrenchTarotEnvironmentSubscriber, \
    ActionResult, ObservationWithGroup, ActionWithGroup
from src.tests.conftest import create_teardown_func, subscriber_receives_data, DummySubscriber


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


def _create_all_phase_agent():
    base_card_neural_net = CoreCardNeuralNet()
    bid_phase_agent_model = BidPhaseAgent.create_dqn(base_card_neural_net)
    bid_phase_agent = BidPhaseAgent(bid_phase_agent_model)
    dog_phase_agent_model = DogPhaseAgent.create_dqn(base_card_neural_net)
    dog_phase_agent = DogPhaseAgent(dog_phase_agent_model)
    return AllPhaseAgent(bid_phase_agent, dog_phase_agent)


def _retrieve_parameter_subset(model):
    return list(model.parameters())[0]
