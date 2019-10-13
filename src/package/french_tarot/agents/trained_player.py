from typing import Dict, Union

from french_tarot.agents.common import CoreCardNeuralNet, Agent, BaseNeuralNetAgent
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.core import Observation
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation
from french_tarot.play_games.datastructures import ModelUpdate


class AllPhaseAgent(Agent):
    _agents: Dict[type, Agent]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_card_neural_net = CoreCardNeuralNet()
        self._agents: Dict[Observation, Union[BaseNeuralNetAgent, Agent]] = {
            BidPhaseObservation: BidPhaseAgent(base_card_neural_net),
            DogPhaseObservation: DogPhaseAgent(base_card_neural_net),
            AnnouncementPhaseObservation: RandomPlayer(),
            CardPhaseObservation: RandomPlayer()
        }

    def get_action(self, observation):
        return self._agents[observation.__class__].get_action(observation)

    def update_model(self, model_update: ModelUpdate):
        type_to_agent_map = {agent.__class__: agent for agent in self._agents.items()}
        for agent_type, new_model in model_update.agent_to_model_map.items():
            type_to_agent_map[agent_type].policy_net.load_state_dict(new_model.state_dict())
