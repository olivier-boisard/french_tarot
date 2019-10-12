from queue import Queue, Empty
from threading import Thread
from typing import Union, List

from attr import dataclass

from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgentTrainer
from french_tarot.environment.core import Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation
from french_tarot.observer import Subscriber, Manager, Message, EventType, Kill


@dataclass
class ActionResult:
    action: any
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
    def __init__(self, bid_phase_trainer: BidPhaseAgentTrainer, dog_phase_trainer: DogPhaseAgentTrainer):
        super().__init__()
        self._training_queue = Queue()
        self._training_thread = Thread(target=self._train)
        self._trainers = {
            BidPhaseObservation: bid_phase_trainer,
            DogPhaseObservation: dog_phase_trainer
        }

    def update(self, action_result: ActionResult):
        self._training_queue.put(action_result)

    def start(self):
        super().start()
        self._training_thread.start()

    def stop(self):
        super().stop()
        self._training_queue.put(Kill())
        self._training_thread.join()

    def _train(self):
        while True:
            for trainer in self._trainers.values():
                trainer.optimize_model()

            try:
                action_result: ActionResult = self._training_queue.get_nowait()
                if not isinstance(action_result, Kill):
                    self._trainers[action_result.observation.__class__].push_to_memory(action_result.observation,
                                                                                       action_result.action,
                                                                                       action_result.reward)
            except Empty:
                pass
