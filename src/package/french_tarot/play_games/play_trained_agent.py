import copy

from tqdm import tqdm

from french_tarot.agents.common import set_all_seeds, CoreCardNeuralNet
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent, DogPhaseAgentTrainer
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation
from french_tarot.observer.core import Message
from french_tarot.observer.managers.abstract_manager import AbstractManager
from french_tarot.observer.managers.event_type import EventType
from french_tarot.observer.managers.manager import Manager
from french_tarot.observer.subscriber import Subscriber, Kill
from french_tarot.play_games.subscriber_wrappers import AllPhaseAgentSubscriber, TrainerSubscriber, \
    FrenchTarotEnvironmentSubscriber, ActionResult, ResetEnvironment


class ActionResultSubscriber(Subscriber):

    def __init__(self, manager: AbstractManager):
        super().__init__(manager)
        self.error = False

    def wait_for_episode_done(self):
        self.loop()

    def loop(self):
        run = True
        while run:
            action_result = self._queue.get()
            if isinstance(action_result, Kill):
                run = False
                self.error = action_result.error
            else:
                run = not action_result.done

    def update(self, data: ActionResult):
        pass

    def dump(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, manager: Manager):
        raise NotImplementedError


def main():
    set_all_seeds()

    steps_per_update = 100
    n_episodes_training: int = 200000
    n_episodes_cold_start = 100
    device = "cuda"

    manager = Manager()
    dog_phase_agent_model = DogPhaseAgent.create_dqn(CoreCardNeuralNet())
    dog_phase_agent = DogPhaseAgent(dog_phase_agent_model)
    agent = AllPhaseAgent(dog_phase_agent=dog_phase_agent)
    agent_subscriber = AllPhaseAgentSubscriber(agent, manager)

    dog_phase_trainer = DogPhaseAgentTrainer(copy.deepcopy(dog_phase_agent_model))
    observation_trainers_map = {DogPhaseObservation: dog_phase_trainer}
    trainer_subscriber = TrainerSubscriber(observation_trainers_map, manager, steps_per_update=steps_per_update,
                                           device=device)

    environment_subscriber = FrenchTarotEnvironmentSubscriber(manager)

    action_subscriber = ActionResultSubscriber(manager)

    manager.add_subscriber(agent_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(agent_subscriber, EventType.MODEL_UPDATE)
    manager.add_subscriber(environment_subscriber, EventType.ACTION)
    manager.add_subscriber(environment_subscriber, EventType.RESET_ENVIRONMENT)
    manager.add_subscriber(trainer_subscriber, EventType.ACTION_RESULT)
    manager.add_subscriber(trainer_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(action_subscriber, EventType.ACTION_RESULT)

    # TODO clean up this mess
    try:
        agent_subscriber.start()
        environment_subscriber.start()
        trainer_subscriber.start()

        # This is necessary to avoid having model update overwhelming trainable agent subscriber, which makes it
        # unable to process observations and prevent training to continue
        print("Wait for initial {} episodes to be complete before proceeding".format(n_episodes_cold_start))
        for _ in tqdm(range(n_episodes_cold_start)):
            action_subscriber.wait_for_episode_done()
            if action_subscriber.error:
                break
            manager.publish(Message(EventType.RESET_ENVIRONMENT, ResetEnvironment()))

        print("Start trainer")
        trainer_subscriber.enable_training()
        for _ in tqdm(range(n_episodes_training)):
            action_subscriber.wait_for_episode_done()
            if action_subscriber.error:
                break
            manager.publish(Message(EventType.RESET_ENVIRONMENT, ResetEnvironment()))

    finally:
        trainer_subscriber.stop()
        environment_subscriber.stop()
        agent_subscriber.stop()


if __name__ == "__main__":
    main()
