from typing import List

import numpy as np
import pandas as pd

from agents.common import Agent, BaseNeuralNetAgent
from agents.random_agent import RandomPlayer
from environment import rotate_list, FrenchTarotEnvironment


def score_diff(agent: Agent, n_episodes_testing=10) -> float:
    random_agent = RandomPlayer()
    agents = [agent, random_agent, random_agent, random_agent]
    all_rewards = []
    for i in range(n_episodes_testing):
        rotation = i % len(agents)
        rotated_agents = rotate_list(agents, rotation)
        rewards = _run_game(FrenchTarotEnvironment(), rotated_agents)
        all_rewards.append(rotate_list(rewards, -rotation))
    mean_scores = pd.DataFrame(all_rewards, columns=["trained", "random_1", "random_2", "random_3"]).mean()
    return mean_scores["trained"] - mean_scores.filter(regex="random_").max()


def _run_game(environment: FrenchTarotEnvironment, agents: List[BaseNeuralNetAgent]) -> List[float]:
    observation = environment.reset()
    done = False
    cnt = 0
    reward = None
    while not done:
        observation, reward, done, _ = environment.step(agents[observation["current_player"]].get_action(observation))
        cnt += 1
        if cnt >= 1000:
            raise RuntimeError("Infinite loop")
    if np.sum(reward) != 0:
        RuntimeError("Scores do not sum up to 0")
    return reward
