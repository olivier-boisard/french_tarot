from typing import Union, List

from attr import dataclass

from french_tarot.agents.trained_player import AllPhaseAgent, AllPhaseTrainer
from french_tarot.environment.core import Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.observer import Subscriber, Manager, Message, EventType


@dataclass
class ActionResult:
    observation: Observation
    reward: Union[float, List[float]]
    done: bool


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


class TrainerSubscriber(Subscriber):
    def __init__(self, trainer: AllPhaseTrainer, batch_size: int = 64):
        super().__init__()
        self.buffer: List[ActionResult] = []
        self._batch_size = batch_size
        self._trainer = trainer

    def update(self, data: ActionResult):
        self.buffer.append(data)
        if len(self.buffer) >= self._batch_size:
            for e in self.buffer:
                self._trainer.push_to_memory(e.observation, e.reward, e.done)
