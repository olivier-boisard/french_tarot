from typing import Dict, Union

from french_tarot.agents.agent import Agent
from french_tarot.agents.bid_phase_agent import BidPhaseAgent
from french_tarot.agents.dog_phase_agent import DogPhaseAgent
from french_tarot.agents.model_update import ModelUpdate
from french_tarot.agents.neural_net_agent import NeuralNetAgent
from french_tarot.agents.random_agent import RandomAgent
from french_tarot.environment.core.core import Observation
from french_tarot.environment.subenvironments.announcements.announcements_phase_observation import \
    AnnouncementsPhaseObservation
from french_tarot.environment.subenvironments.bid.bid_phase_observation import BidPhaseObservation
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation
from french_tarot.environment.subenvironments.dog.dog_phase_observation import DogPhaseObservation


class AllPhaseAgent(Agent):
    _agents: Dict[type, Agent]

    def __init__(self, bid_phase_agent: BidPhaseAgent = None, dog_phase_agent: DogPhaseAgent = None, **kwargs):
        super().__init__(**kwargs)
        self._agents: Dict[Observation, Union[NeuralNetAgent, Agent]] = {
            BidPhaseObservation: RandomAgent() if bid_phase_agent is None else bid_phase_agent,
            DogPhaseObservation: RandomAgent() if dog_phase_agent is None else dog_phase_agent,
            AnnouncementsPhaseObservation: RandomAgent(),
            CardPhaseObservation: RandomAgent()
        }

    def get_action(self, observation):
        return self._agents[observation.__class__].get_action(observation)

    def update_model(self, model_update: ModelUpdate):
        # TODO replace isinstance with polymorphism
        neural_net_agents = filter(lambda agent: isinstance(agent, NeuralNetAgent), self._agents.values())
        type_to_agent = {agent.policy_net.__class__: agent for agent in neural_net_agents}
        for new_model in model_update.models:
            type_to_agent[new_model.__class__].update_policy_net(new_model)
