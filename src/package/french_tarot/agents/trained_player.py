from typing import Dict

from french_tarot.agents.common import CoreCardNeuralNet, Agent
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


class AllPhaseAgent(Agent):
    _agents: Dict[type, Agent]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_per_phase_agents()

    def _initialize_per_phase_agents(self):
        base_card_neural_net = CoreCardNeuralNet()
        self._agents = {
            BidPhaseObservation: BidPhaseAgent(base_card_neural_net),
            DogPhaseObservation: DogPhaseAgent(base_card_neural_net),
            AnnouncementPhaseObservation: RandomPlayer(),
            CardPhaseObservation: RandomPlayer()
        }

    def get_action(self, observation):
        return self._agents[observation.__class__].get_action(observation)
