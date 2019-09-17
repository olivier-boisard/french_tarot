import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from french_tarot.agents.common import core, CoreCardNeuralNet, Agent
from french_tarot.agents.meta import singledispatchmethod
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_bid import BidPhaseAgent
from french_tarot.agents.trained_player_dog import DogPhaseAgent
from french_tarot.environment.common import CARDS
from french_tarot.environment.observations import Observation, BidPhaseObservation, DogPhaseObservation, \
    AnnouncementPhaseObservation, CardPhaseObservation


class TrainedPlayer(Agent):

    def __init__(self, bid_phase_agent: torch.nn.Module = None, dog_phase_agent: torch.nn.Module = None,
                 summary_writer: SummaryWriter = None):
        random_agent = RandomPlayer()
        base_card_neural_net = CoreCardNeuralNet()
        self._agents = {
            BidPhaseObservation.__name__: BidPhaseAgent(
                base_card_neural_net, summary_writer=summary_writer) if bid_phase_agent is None else bid_phase_agent,
            DogPhaseObservation.__name__: DogPhaseAgent(
                base_card_neural_net, summary_writer=summary_writer) if dog_phase_agent is None else dog_phase_agent,
            AnnouncementPhaseObservation.__name__: random_agent,  # trained agent not implemented yet
            CardPhaseObservation.__name__: random_agent  # trained agent not implemented yet
        }

    def optimize_model(self):
        for model in self._agents.values():
            model.optimize_model()

    def get_action(self, observation: Observation):
        return self._agents[observation.__class__.__name__].get_action(observation)

    @singledispatchmethod
    def push_to_agent_memory(self, observation: Observation, action, reward: float):
        raise ValueError("Unhandled observation type")

    @push_to_agent_memory.register
    def _(self, observation: DogPhaseObservation, action, reward):
        selected_cards = torch.zeros(len(CARDS))
        for permuted_action in itertools.permutations(action):
            hand = list(observation.hand)
            for card in permuted_action:
                xx = torch.cat((core(hand), selected_cards)).unsqueeze(0)
                action_id = DogPhaseAgent.CARDS_OK_IN_DOG.index(card)
                self._agents[DogPhaseObservation.__name__].memory.push(xx, action_id, None, reward)

                selected_cards[action_id] = 1
                hand.remove(DogPhaseAgent.CARDS_OK_IN_DOG[action_id])

    @push_to_agent_memory.register
    def _(self, observation: BidPhaseObservation, action, reward):
        self._agents[BidPhaseObservation.__name__].memory.push(
            core(observation.hand).unsqueeze(0),
            action,
            None,
            reward
        )
