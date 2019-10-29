import copy
from abc import ABC
from queue import Queue, Empty
from threading import Thread
from typing import Union, List, Dict

import dill
from attr import dataclass

from french_tarot.agents.common import Trainer
from french_tarot.agents.meta import singledispatchmethod
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.core import Observation
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation
from french_tarot.exceptions import FrenchTarotException
from french_tarot.observer.core import Message
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from french_tarot.observer.subscriber import Subscriber, Kill
from french_tarot.play_games.datastructures import ModelUpdate


@dataclass
class WithGroupBaseClass(ABC):
    observation_action_reward_group: int


@dataclass
class ActionResult(WithGroupBaseClass):
    action: any
    next_observation: Observation
    reward: Union[float, List[float]]
    done: bool


@dataclass
class ObservationWithGroup(WithGroupBaseClass):
    observation: Observation


@dataclass
class ActionWithGroup(WithGroupBaseClass):
    action: any


class ResetEnvironment:
    pass


class AllPhaseAgentSubscriber(Subscriber):
    def __init__(self, agent: AllPhaseAgent, manager: Manager):
        super().__init__(manager)
        self._manager: Manager = manager
        self._agent = agent

    @singledispatchmethod
    def update(self, data):
        if data is not None:
            raise FrenchTarotException("data should be None if not of supported types")

    @update.register
    def _(self, observation_with_group: ObservationWithGroup):
        action = self._agent.get_action(observation_with_group.observation)
        self._manager.publish(Message(EventType.ACTION,
                                      ActionWithGroup(observation_with_group.observation_action_reward_group, action)))

    @update.register
    def _(self, model_update: ModelUpdate):
        self._agent.update_model(model_update)

    def dump(self, path: str):
        with open(path, "wb") as f:
            dill.dump(self._agent, f)

    @classmethod
    def load(cls, path: str, manager: Manager):
        with open(path, "rb") as f:
            agent = dill.load(f)
        return cls(agent, manager)


class FrenchTarotEnvironmentSubscriber(Subscriber):

    def __init__(self, manager: Manager):
        super().__init__(manager)
        self._environment = FrenchTarotEnvironment()
        self._manager: Manager = manager

    def setup(self):
        observation = self._environment.reset()
        self._manager.publish(Message(EventType.OBSERVATION, ObservationWithGroup(0, observation)))

    @singledispatchmethod
    def update(self, data: any):
        raise NotImplementedError

    @update.register
    def _(self, action_with_group: ActionWithGroup):
        observation, reward, done, _ = self._environment.step(action_with_group.action)
        new_group = action_with_group.observation_action_reward_group + 1
        action_result = ActionResult(new_group, action_with_group, observation, reward, done)
        self._manager.publish(Message(EventType.ACTION_RESULT, action_result))
        if not done:
            self._manager.publish(Message(EventType.OBSERVATION, ObservationWithGroup(new_group, observation)))

    @update.register
    def _(self, _: ResetEnvironment):
        self.setup()

    def dump(self, path: str):
        with open(path, "wb") as f:
            dill.dump(self._environment, f)

    @classmethod
    def load(cls, path: str, manager: Manager):
        with open(path, "rb") as f:
            environment = dill.load(f)
        obj = cls(manager)
        obj._environment = environment
        return obj


class TrainerSubscriber(Subscriber):

    def __init__(self, observation_trainers_map: Dict[type, Trainer], manager: Manager, steps_per_update: int = 100):
        super().__init__(manager)
        self._pre_card_phase_observations_and_action_results = []
        self._training_queue = Queue()
        self._training_thread = Thread(target=self._train)
        self._steps_per_update = steps_per_update
        self._manager = manager
        self._action_results = {}
        self._observations = {}
        self._trainers = observation_trainers_map

    @singledispatchmethod
    def update(self, data: any):
        pass

    @update.register
    def _(self, action_result: ActionResult):
        self._action_results[action_result.observation_action_reward_group] = action_result
        self._match_action_results_and_observation()
        if action_result.done:
            self._push_early_actions_to_replay_memory()

    @update.register
    def _(self, observation_with_group: ObservationWithGroup):
        self._observations[observation_with_group.observation_action_reward_group] = observation_with_group
        self._match_action_results_and_observation()

    def dump(self, path: str):
        pass

    @classmethod
    def load(cls, path: str, manager: Manager):
        pass

    def _match_action_results_and_observation(self):
        keys = filter(lambda k: k in self._observations, self._action_results.keys())
        for key in keys:
            observation = self._observations.pop(key)
            action_result = self._action_results.pop(key)
            observation_and_action_results = (observation, action_result)
            if not isinstance(observation, CardPhaseObservation):
                self._pre_card_phase_observations_and_action_results.append(observation_and_action_results)
            else:
                self._training_queue.put(observation_and_action_results)

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
                    observation, action_result = new_entry
                    self._trainers[observation.__class__].push_to_memory(observation, action_result.action,
                                                                         action_result.reward)
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

    def _push_early_actions_to_replay_memory(self):
        for observation, registered_action_result in self._pre_card_phase_observations_and_action_results:
            action_result_with_updated_reward = copy.copy(registered_action_result)
            index = observation.player.position_towards_taker
            action_result_with_updated_reward.reward = registered_action_result.reward[index]
            self._training_queue.put((observation, action_result_with_updated_reward))
