from french_tarot.agents.trained_player import AllPhasePlayerTrainer
from french_tarot.observer import Subscriber, Manager, Message, EventType


class AgentSubscriber(Subscriber):
    def __init__(self, manager: Manager):
        super().__init__()
        self._manager: Manager = manager
        self._agent = AllPhasePlayerTrainer()

    def update(self, observation: any):
        action = self._agent.get_action(observation)
        self._manager.publish(Message(EventType.ACTION, action))
