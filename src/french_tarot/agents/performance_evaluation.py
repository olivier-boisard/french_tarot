from typing import List

import numpy as np
import pandas as pd

from french_tarot.agents.common import Agent, BaseNeuralNetAgent
from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.core import rotate_list


def compute_diff_score_metric(agent: Agent, n_episodes_testing=10) -> float:
    random_agent = RandomPlayer()
    players = [agent, random_agent, random_agent, random_agent]
    round_scores = _play_games(players, n_episodes_testing)
    mean_scores = round_scores.mean()
    score = mean_scores["trained"] - mean_scores.filter(regex="random_").max()
    return score


def _play_games(agents, n_games):
    round_scores = []
    for i in range(n_games):
        rotation = i % len(agents)
        rotated_agents = rotate_list(agents, rotation)
        rewards = _run_game(FrenchTarotEnvironment(), rotated_agents)
        round_scores.append(rotate_list(rewards, -rotation))
    round_scores = pd.DataFrame(round_scores, columns=["trained", "random_1", "random_2", "random_3"])
    return round_scores


def _run_game(environment: FrenchTarotEnvironment, players: List[BaseNeuralNetAgent]) -> List[float]:
    observation = environment.reset()
    done = False
    cnt = 0
    reward = None
    while not done:
        current_player = players[observation.current_player_id]
        observation, reward, done, _ = environment.step(current_player.get_action(observation))
        cnt += 1
        if cnt >= 1000:
            raise RuntimeError("Infinite loop")
    if np.sum(reward) != 0:
        RuntimeError("Scores do not sum up to 0")
    return reward
