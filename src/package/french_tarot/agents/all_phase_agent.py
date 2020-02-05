from typing import Dict, Union

from french_tarot.agents.agent import Agent, ActionWithProbability
from french_tarot.agents.card_phase_agent import CardPhaseAgent
from french_tarot.agents.model_update import ModelUpdate
from french_tarot.agents.random_agent import RandomAgent
from french_tarot.environment.core.core import Observation
from french_tarot.environment.subenvironments.announcements.announcements_phase_observation import \
    AnnouncementsPhaseObservation
from french_tarot.environment.subenvironments.bid.bid_phase_observation import BidPhaseObservation
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation
from french_tarot.environment.subenvironments.dog.dog_phase_observation import DogPhaseObservation


class AllPhaseAgent(Agent):
    _agents: Dict[type, Agent]

    def __init__(self, card_phase_agent: CardPhaseAgent = None):
        super().__init__()
        self._agents: Dict[Observation, Union[Agent, CardPhaseAgent]] = {
            BidPhaseObservation: RandomAgent(),
            DogPhaseObservation: RandomAgent(),
            AnnouncementsPhaseObservation: RandomAgent(),
            CardPhaseObservation: card_phase_agent if card_phase_agent is not None else RandomAgent()
        }

    def get_action(self, observation: Observation) -> ActionWithProbability:
        return self._agents[observation.__class__].get_action(observation)

    def update_model(self, model_update: ModelUpdate):
        # TODO replace isinstance with polymorphism
        neural_net_agents = filter(lambda agent: isinstance(agent, CardPhaseObservation), self._agents.values())
        type_to_agent = {agent.policy_net.__class__: agent for agent in neural_net_agents}
        for new_model in model_update.models:
            type_to_agent[new_model.__class__].update_policy_net(new_model)
