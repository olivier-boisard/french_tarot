from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer import Subscriber, Manager, Message, EventType


class ActionResult:
    pass


class AgentSubscriber(Subscriber):
    def __init__(self, manager: Manager):
        super().__init__()
        self._manager: Manager = manager
        self._agent = AllPhaseAgent()

    def update(self, observation: any):
        action = self._agent.get_action(observation)
        self._manager.publish(Message(EventType.ACTION, action))


class FrenchTarotEnvironmentSubscriber(Subscriber):

    def __init__(self, manager: Manager):
        super().__init__()
        self._environment = FrenchTarotEnvironment()
        self._manager: Manager = manager

    def setup(self):
        observation = self._environment.reset()
        self._manager.publish(Message(EventType.OBSERVATION, observation))

    def update(self, action: any):
        observation, _, _, _ = self._environment.step(action)
        self._manager.publish(Message(EventType.OBSERVATION, observation))
        self._manager.publish(Message(EventType.ACTION_RESULT, ActionResult()))
