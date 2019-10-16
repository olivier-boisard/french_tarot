import copy
from queue import Queue, Empty
from threading import Thread
from typing import Union, List

from attr import dataclass

from french_tarot.agents.meta import singledispatchmethod
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgentTrainer, BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgentTrainer, DogPhaseAgent
from french_tarot.environment.core import Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation
from french_tarot.exceptions import FrenchTarotException
from french_tarot.observer import Subscriber, Manager, Message, EventType, Kill
from french_tarot.play_games.datastructures import ModelUpdate


@dataclass
class ActionResult:
    action: any
    next_observation: Observation
    reward: Union[float, List[float]]
    done: bool


class ResetEnvironment:
    pass


class AllPhaseAgentSubscriber(Subscriber):
    def __init__(self, agent: AllPhaseAgent, manager: Manager):
        super().__init__()
        self._manager: Manager = manager
        self._agent = agent

    @singledispatchmethod
    def update(self, data, *_):
        if data is not None:
            raise FrenchTarotException("data should be None if not of supported types")

    @update.register
    def _(self, observation: Observation, group: int):
        action = self._agent.get_action(observation)
        self._manager.publish(Message(EventType.ACTION, action, group=group))

    @update.register
    def _(self, model_update: ModelUpdate, *_):
        self._agent.update_model(model_update)


class FrenchTarotEnvironmentSubscriber(Subscriber):

    def __init__(self, manager: Manager):
        super().__init__()
        self._environment = FrenchTarotEnvironment()
        self._manager: Manager = manager

    def setup(self):
        observation = self._environment.reset()
        self._manager.publish(Message(EventType.OBSERVATION, observation, group=0))

    @singledispatchmethod
    def update(self, action, group: int):
        observation, reward, done, _ = self._environment.step(action)
        self._manager.publish(Message(EventType.ACTION_RESULT, ActionResult(action, observation, reward, done),
                                      group=group))
        self._manager.publish(Message(EventType.OBSERVATION, observation, group=group + 1))

    @update.register
    def _(self, _: ResetEnvironment, *__):
        self.setup()


class TrainerSubscriber(Subscriber):
    def __init__(self, bid_phase_trainer: BidPhaseAgentTrainer, dog_phase_trainer: DogPhaseAgentTrainer,
                 manager: Manager, steps_per_update: int = 100):
        super().__init__()
        self._training_queue = Queue()
        self._training_thread = Thread(target=self._train)
        self._steps_per_update = steps_per_update
        self._manager = manager
        self._action_results = {}
        self._observations = {}
        self._trainers = {
            BidPhaseObservation: bid_phase_trainer,
            DogPhaseObservation: dog_phase_trainer
        }

    @singledispatchmethod
    def update(self, *_):
        pass

    @update.register
    def _(self, action_result: ActionResult, group_id: int):
        self._action_results[group_id] = action_result
        self._match_action_results_and_observation()

    @update.register
    def _(self, observation: Observation, group_id: int):
        self._observations[group_id] = observation
        self._match_action_results_and_observation()

    def _match_action_results_and_observation(self):
        keys = filter(lambda k: k in self._observations, self._action_results.keys())
        for key in keys:
            observation = self._observations.pop(key)
            action_result = self._action_results.pop(key)
            new_entry = (observation, action_result.action, action_result.reward, action_result.next_observation)
            self._training_queue.put(new_entry)

    def start(self):
        super().start()
        self._training_thread.start()

    def stop(self):
        self._training_queue.put(Kill())
        super().stop()
        self._training_thread.join()

    def _train(self):
        step = 0
        while True:
            step += 1
            for trainer in self._trainers.values():
                trainer.optimize_model()

            try:
                new_entry = self._training_queue.get_nowait()
                if not isinstance(new_entry, Kill):
                    observation, action, reward, next_observation = new_entry
                    self._trainers[next_observation.__class__].push_to_memory(observation, action, reward)
                else:
                    break
                self._training_queue.task_done()
            except Empty:
                pass

            if step % self._steps_per_update == 0:
                new_models = ModelUpdate({
                    BidPhaseAgent: copy.deepcopy(self._trainers[BidPhaseObservation].model),
                    DogPhaseAgent: copy.deepcopy(self._trainers[DogPhaseObservation].model)
                })
                self._manager.publish(Message(EventType.MODEL_UPDATE, new_models))
