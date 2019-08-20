from agents.common import card_set_encoder
from agents.random_agent import RandomPlayer
from agents.trained_player_bid import BidPhaseAgent
from agents.trained_player_dog import DogPhaseAgent
from environment import GamePhase


class TrainedPlayer:

    def __init__(self, bid_phase_agent=None, dog_phase_agent=None):
        random_agent = RandomPlayer()
        self._agents = {
            GamePhase.BID: BidPhaseAgent() if bid_phase_agent is None else bid_phase_agent,
            GamePhase.DOG: DogPhaseAgent() if dog_phase_agent is None else dog_phase_agent,
            GamePhase.ANNOUNCEMENTS: random_agent,  # trained agent not implemented yet
            GamePhase.CARD: random_agent  # trained agent not implemented yet
        }

    def get_action(self, observation):
        return self._agents[observation["game_phase"]].get_action(observation)

    def push_to_agent_memory(self, observation, action, reward):
        self._agents[observation["game_phase"]].memory.push(card_set_encoder(observation).unsqueeze(0), action.value,
                                                            None, reward)
