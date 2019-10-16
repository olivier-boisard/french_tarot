from tqdm import tqdm

from french_tarot.agents.common import set_all_seeds, CoreCardNeuralNet
from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.agents.trained_player_bid import BidPhaseAgent, BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgent, DogPhaseAgentTrainer
from french_tarot.observer import Manager, Subscriber, EventType, Message
from french_tarot.play_games.subscriber_wrappers import AllPhaseAgentSubscriber, TrainerSubscriber, \
    FrenchTarotEnvironmentSubscriber, ActionResult, ResetEnvironment


class ActionResultSubscriber(Subscriber):

    def __init__(self):
        super().__init__()

    def wait_for_episode_done(self):
        self.loop()

    def loop(self):
        run = True
        while run:
            action_result: ActionResult = self._queue.get()
            run = not action_result.done
            self._queue.task_done()

    def update(self, data: ActionResult):
        pass


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

    action_subscriber = ActionResultSubscriber()

    manager.add_subscriber(agent_subscriber, EventType.OBSERVATION)
    manager.add_subscriber(agent_subscriber, EventType.MODEL_UPDATE)
    manager.add_subscriber(environment_subscriber, EventType.ACTION)
    manager.add_subscriber(trainer_subscriber, EventType.ACTION_RESULT)
    manager.add_subscriber(action_subscriber, EventType.ACTION_RESULT)

    try:
        agent_subscriber.start()
        trainer_subscriber.start()
        environment_subscriber.start()
        for _ in tqdm(range(n_episodes_training)):
            action_subscriber.wait_for_episode_done()
            manager.publish(Message(EventType.RESET_ENVIRONMENT, ResetEnvironment()))
    finally:
        agent_subscriber.stop()
        trainer_subscriber.stop()
        environment_subscriber.stop()


if __name__ == "__main__":
    main()
