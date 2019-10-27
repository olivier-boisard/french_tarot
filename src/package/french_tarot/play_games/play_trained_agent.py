import datetime

from tqdm import tqdm

from french_tarot.agents.common import set_all_seeds, CoreCardNeuralNet
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgent, BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgent, DogPhaseAgentTrainer
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
            self._queue.task_done()

    def update(self, data: ActionResult):
        pass

    def dump(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, manager: Manager):
        raise NotImplementedError


def main(n_episodes_training: int = 200000):
    set_all_seeds()

    steps_per_update = 100

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

    environment_subscriber = FrenchTarotEnvironmentSubscriber(manager)

    action_subscriber = ActionResultSubscriber(manager)

    manager.add_subscriber(agent_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(agent_subscriber, EventType.MODEL_UPDATE)
    manager.add_subscriber(environment_subscriber, EventType.ACTION)
    manager.add_subscriber(environment_subscriber, EventType.RESET_ENVIRONMENT)
    manager.add_subscriber(trainer_subscriber, EventType.ACTION_RESULT)
    manager.add_subscriber(action_subscriber, EventType.ACTION_RESULT)

    try:
        agent_subscriber.start()
        trainer_subscriber.start()
        environment_subscriber.start()
        for _ in tqdm(range(n_episodes_training)):
            action_subscriber.wait_for_episode_done()
            if action_subscriber.error:
                timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S")
                output_path = "history_" + timestamp + ".dill"
                print("Dumping history at", output_path)
                manager.dump_history(output_path)
                break
            manager.publish(Message(EventType.RESET_ENVIRONMENT, ResetEnvironment()))
    finally:
        agent_subscriber.stop()
        trainer_subscriber.stop()
        environment_subscriber.stop()


if __name__ == "__main__":
    main()
