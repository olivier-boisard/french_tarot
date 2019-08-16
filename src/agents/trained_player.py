from agents.common import card_set_encoder
from agents.random_agent import RandomPlayer
from agents.trained_player_bid import BidPhaseAgent
from agents.trained_player_dog import DogPhaseAgent
from environment import GamePhase


class TrainedPlayer:

    def __init__(self):
        random_agent = RandomPlayer()
        self._agents = {
            GamePhase.BID: BidPhaseAgent(),
            GamePhase.DOG: DogPhaseAgent(),
            GamePhase.ANNOUNCEMENTS: random_agent,  # trained agent not implemented yet
            GamePhase.CARD: random_agent  # trained agent not implemented yet
        }

    def get_action(self, observation):
        return self._agents[observation["game_phase"]].get_action(observation)

    def push_to_agent_memory(self, observation, action, reward):
        if observation["game_phase"] == GamePhase.BID:
            reward_scaling_factor = 100.
            self._agents[GamePhase.BID].push(card_set_encoder(observation).unsqueeze(0), action.value, None,
                                             reward / reward_scaling_factor)
        elif observation["game_phase"] == GamePhase.DOG:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
