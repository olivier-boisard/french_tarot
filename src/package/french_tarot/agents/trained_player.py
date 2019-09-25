from abc import ABC
from typing import Union, Dict

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from french_tarot.agents.common import CoreCardNeuralNet, Agent, Trainer
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent, BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgent, DogPhaseAgentTrainer
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


class DummyTrainer:

    def optimize_model(self, *args, **kwargs):
        pass


class AgentWithTrainer(ABC):
    def __init__(self, agent: Agent, trainer: Union[Trainer, DummyTrainer]):
        self.agent = agent
        self.trainer = trainer

    def optimize_model(self):
        self.trainer.optimize_model()


class RandomAgentWithDummyTrainer(AgentWithTrainer):
    def __init__(self, agent):
        super().__init__(agent, DummyTrainer())


class BidPhaseAgentWithTrainer(AgentWithTrainer):
    def __init__(self, base_card_neural_net: nn.Module, summary_writer: SummaryWriter = None):
        agent = BidPhaseAgent(base_card_neural_net)
        trainer = BidPhaseAgentTrainer(agent.policy_net, summary_writer=summary_writer, name="bid")
        super().__init__(agent, trainer)


class DogPhaseAgentWithTrainer(AgentWithTrainer):
    def __init__(self, base_card_neural_net: nn.Module, summary_writer: SummaryWriter = None):
        agent = DogPhaseAgent(base_card_neural_net)
        trainer = DogPhaseAgentTrainer(agent.policy_net, summary_writer=summary_writer, name="dog")
        super().__init__(agent, trainer)


class AllPhasePlayerTrainer(Agent):
    _agents_with_trainers: Dict[type, AgentWithTrainer]

    def __init__(self, summary_writer: SummaryWriter = None, **kwargs):
        super().__init__(**kwargs)
        self._initialize_per_phase_agents(summary_writer)

    def _initialize_per_phase_agents(self, summary_writer):
        base_card_neural_net = CoreCardNeuralNet()
        self._agents_with_trainers = {
            BidPhaseObservation: BidPhaseAgentWithTrainer(base_card_neural_net, summary_writer=summary_writer),
            DogPhaseObservation: DogPhaseAgentWithTrainer(base_card_neural_net, summary_writer=summary_writer),
            AnnouncementPhaseObservation: RandomAgentWithDummyTrainer(RandomPlayer()),
            CardPhaseObservation: RandomAgentWithDummyTrainer(RandomPlayer())
        }

    def optimize_model(self):
        for model in self._agents_with_trainers.values():
            model.optimize_model()

    def get_action(self, observation):
        return self._agents_with_trainers[observation.__class__].agent.get_action(observation)

    def push_to_agent_memory(self, observation, action, reward):
        self._agents_with_trainers[observation.__class__].trainer.push_to_memory(observation, action, reward)
