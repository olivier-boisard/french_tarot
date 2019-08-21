import numpy as np
import torch
import tqdm

from agents.trained_player import TrainedPlayer
from environment import FrenchTarotEnvironment, GamePhase, rotate_list, Bid


def _main():
    _set_all_seeds()
    agent = TrainedPlayer()
    all_rewards = _run_training(agent)
    _display_results(all_rewards)


def _display_results(all_rewards):
    raise NotImplementedError()


def _run_training(agent, n_iterations=200000):
    environment = FrenchTarotEnvironment()
    all_rewards = []
    for i in tqdm.tqdm(range(n_iterations)):
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

        all_rewards.append(np.roll(rewards, i % environment.n_players))
        agent.optimize_models()
    return all_rewards


def _set_all_seeds(seed=1988):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    _main()
