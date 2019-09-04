import itertools

import torch

from agents.common import card_set_encoder, BaseCardNeuralNet, Agent
from agents.random_agent import RandomPlayer
from agents.trained_player_bid import BidPhaseAgent
from agents.trained_player_dog import DogPhaseAgent
from environment import GamePhase, Card


class TrainedPlayer(Agent):

    def optimize_model(self):
        for model in self._agents.values():
            model.optimize_model()

    def __init__(self, bid_phase_agent: torch.nn.Module = None, dog_phase_agent: torch.nn.Module = None):
        random_agent = RandomPlayer()
        base_card_neural_net = BaseCardNeuralNet()
        self._agents = {
            GamePhase.BID: BidPhaseAgent(base_card_neural_net) if bid_phase_agent is None else bid_phase_agent,
            GamePhase.DOG: DogPhaseAgent(base_card_neural_net) if dog_phase_agent is None else dog_phase_agent,
            GamePhase.ANNOUNCEMENTS: random_agent,  # trained agent not implemented yet
            GamePhase.CARD: random_agent  # trained agent not implemented yet
        }

    def get_action(self, observation: dict):
        return self._agents[observation["game_phase"]].get_action(observation)

    def push_to_agent_memory(self, observation: dict, action, reward: float):
        if observation["game_phase"] == GamePhase.BID:
            self._agents[GamePhase.BID].memory.push(card_set_encoder(observation["hand"]).unsqueeze(0),
                                                    action,
                                                    None, reward)
        elif observation["game_phase"] == GamePhase.DOG:
            # noinspection PyTypeChecker
            selected_cards = torch.zeros(len(list(Card)))
            for permuted_action in itertools.permutations(action):
                hand = list(observation["hand"])
                for card in permuted_action:
                    xx = torch.cat((card_set_encoder(hand), selected_cards)).unsqueeze(0)
                    action_id = DogPhaseAgent.CARDS_OK_IN_DOG.index(card)
                    self._agents[GamePhase.DOG].memory.push(xx, action_id, None, reward)

                    selected_cards[action_id] = 1
                    hand.remove(DogPhaseAgent.CARDS_OK_IN_DOG[action_id])
        else:
            raise NotImplemented()
