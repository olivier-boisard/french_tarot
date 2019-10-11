from typing import Dict

from french_tarot.agents.common import CoreCardNeuralNet, Agent
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent, BidPhaseAgentTrainer
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


class AllPhaseAgent(Agent):
    _agents: Dict[type, Agent]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_card_neural_net = CoreCardNeuralNet()
        self._agents = {
            BidPhaseObservation: BidPhaseAgent(base_card_neural_net),
            DogPhaseObservation: DogPhaseAgent(base_card_neural_net),
            AnnouncementPhaseObservation: RandomPlayer(),
            CardPhaseObservation: RandomPlayer()
        }

    def get_action(self, observation):
        return self._agents[observation.__class__].get_action(observation)


class AllPhaseTrainer:
    def __init__(self, bid_phase_trainer: BidPhaseAgentTrainer):
        self._bid_phase_trainer = bid_phase_trainer

    def push_to_memory(self, observation, reward, done):
        self._bid_phase_trainer.push_to_memory(observation, reward, done)
