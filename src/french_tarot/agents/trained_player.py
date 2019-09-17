from torch.utils.tensorboard import SummaryWriter

from french_tarot.agents.common import CoreCardNeuralNet, Agent
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent, BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgent, DogPhaseAgentTrainer
from french_tarot.environment.observations import Observation, BidPhaseObservation, DogPhaseObservation, \
    AnnouncementPhaseObservation, CardPhaseObservation


class AllPhasePlayerTrainer(Agent):

    def __init__(self, summary_writer: SummaryWriter = None, **kwargs):
        super().__init__(**kwargs)
        self._initialize_per_phase_agents()
        self._initialize_trainers(summary_writer)

    def _initialize_per_phase_agents(self):
        base_card_neural_net = CoreCardNeuralNet()
        self._agents = {
            BidPhaseObservation: BidPhaseAgent(base_card_neural_net),
            DogPhaseObservation: DogPhaseAgent(base_card_neural_net),
            AnnouncementPhaseObservation: RandomPlayer(),
            CardPhaseObservation: RandomPlayer()
        }

    def _initialize_trainers(self, summary_writer):
        self._trainers = {
            BidPhaseObservation: BidPhaseAgentTrainer(
                self._agents[BidPhaseObservation].policy_net,
                summary_writer=summary_writer
            ),
            DogPhaseObservation: DogPhaseAgentTrainer(
                self._agents[DogPhaseObservation].policy_net,
                summary_writer=summary_writer
            )
        }

    def optimize_model(self):
        for model in self._trainers.values():
            model.optimize_model()

    def get_action(self, observation: Observation):
        return self._agents[observation.__class__].get_action(observation)
