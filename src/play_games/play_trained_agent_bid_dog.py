import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

from agents.common import set_all_seeds
from agents.performance_evaluation import evaluate_agent_performance
from agents.trained_player import TrainedPlayer
from environment import FrenchTarotEnvironment, GamePhase, rotate_list, Bid


def _main(n_episodes_training: int = 200000, n_episodes_testing: int = 1000):
    set_all_seeds()
    writer = SummaryWriter()
    trained_agent = TrainedPlayer()
    _run_training(trained_agent, n_episodes_training, writer)
    all_rewards = evaluate_agent_performance(trained_agent, n_episodes_testing)
    print("Scores per agent:", all_rewards.mean())
    all_rewards.to_csv("bid_dog_results.csv")
    all_rewards.plot()


def _run_training(agent: TrainedPlayer, n_episodes: int, tb_writer: SummaryWriter):
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

        agent.optimize_model(tb_writer)


if __name__ == "__main__":
    _main()
