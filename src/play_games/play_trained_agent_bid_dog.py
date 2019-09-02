from typing import List

import numpy as np
import pandas as pd
import tqdm

from agents.common import TrainableAgent, set_all_seeds
from agents.random_agent import RandomPlayer
from agents.trained_player import TrainedPlayer
from environment import FrenchTarotEnvironment, GamePhase, rotate_list, Bid


def _main(n_episodes_training: int = 200000, n_episodes_testing: int = 1000):
    set_all_seeds()
    trained_agent = TrainedPlayer()
    _run_training(trained_agent, n_episodes_training)

    random_agent = RandomPlayer()
    agents = [trained_agent, random_agent, random_agent, random_agent]
    all_rewards = []
    for i in range(n_episodes_testing):
        rotation = i % len(agents)
        rotated_agents = rotate_list(agents, rotation)
        rewards = _run_game(FrenchTarotEnvironment(), rotated_agents)
        all_rewards.append(rotate_list(rewards, -rotation))
    all_rewards = pd.DataFrame(all_rewards, columns=["trained", "random_1", "random_2", "random_3"])
    all_rewards.to_csv("big_dog_results.csv")
    all_rewards.plot()


def _run_training(agent: TrainedPlayer, n_episodes: int):
    environment = FrenchTarotEnvironment()
    for _ in tqdm.tqdm(range(n_episodes)):
        observation = environment.reset()
        done = False

        early_phases_observations = []
        early_phases_actions = []
        rewards = None
        while not done:
            action = agent.get_action(observation)
            new_observation, rewards, done, _ = environment.step(action)

            if observation["game_phase"] <= GamePhase.DOG:
                early_phases_observations.append(observation)
                early_phases_actions.append(action)
            else:
                pass  # Nothing to do
            observation = new_observation

        if rewards is None:
            raise RuntimeError("No rewards set")
        rewards = rotate_list(rewards, environment.taking_player_original_id)
        # Add reward of creating the dog, i.e. append the reward that we got at the end of the game for the taker
        max_bid = np.max(observation["bid_per_player"])
        if Bid.PASS < max_bid < Bid.GARDE_SANS:
            rewards.append(rewards[environment.taking_player_original_id])
        assert len(rewards) == len(early_phases_observations)
        assert len(rewards) == len(early_phases_actions)
        for observation, action, reward in zip(early_phases_observations, early_phases_actions, rewards):
            agent.push_to_agent_memory(observation, action, reward)

        agent.optimize_models()


def _run_game(environment: FrenchTarotEnvironment, agents: List[TrainableAgent]) -> List[float]:
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


if __name__ == "__main__":
    _main()
