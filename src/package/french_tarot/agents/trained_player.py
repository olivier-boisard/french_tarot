from dataclasses import dataclass
from typing import Dict, Union, List

from french_tarot.agents.agent import Agent
from french_tarot.agents.neural_net import BaseNeuralNetAgent
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.core import Observation
from french_tarot.environment.subenvironments.announcements_phase import AnnouncementPhaseObservation
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


class AllPhaseAgent(Agent):
    _agents: Dict[type, Agent]

    def __init__(self, bid_phase_agent: BidPhaseAgent = None, dog_phase_agent: DogPhaseAgent = None, **kwargs):
        super().__init__(**kwargs)
        self._agents: Dict[Observation, Union[BaseNeuralNetAgent, Agent]] = {
            BidPhaseObservation: RandomPlayer() if bid_phase_agent is None else bid_phase_agent,
            DogPhaseObservation: RandomPlayer() if dog_phase_agent is None else dog_phase_agent,
            AnnouncementPhaseObservation: RandomPlayer(),
            CardPhaseObservation: RandomPlayer()
        }

    def get_action(self, observation):
        return self._agents[observation.__class__].get_action(observation)

    def update_model(self, model_update: 'ModelUpdate'):
        neural_net_agents = filter(lambda agent: isinstance(agent, BaseNeuralNetAgent), self._agents.values())
        type_to_agent = {agent.policy_net.__class__: agent for agent in neural_net_agents}
        for new_model in model_update.models:
            type_to_agent[new_model.__class__].update_policy_net(new_model)


@dataclass
class ModelUpdate:
    models: List[Dict]
