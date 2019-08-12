import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from agents.random_agent import RandomPlayer
from agents.trained_player import BidPhaseAgent, bid_phase_observation_encoder
from environment import FrenchTarotEnvironment, GamePhase, rotate_list


def _main():
    _set_all_seeds()
    bid_phase_dqn_agent = BidPhaseAgent()
    all_rewards = _run_training(bid_phase_dqn_agent)

    dump_and_display_results(all_rewards, bid_phase_dqn_agent.loss)


def _run_training(bid_phase_dqn_agent, n_iterations=200000):
    environment = FrenchTarotEnvironment()
    random_agent = RandomPlayer()
    all_rewards = []
    for i in tqdm.tqdm(range(n_iterations)):
        observation = environment.reset()
        done = False
        observations_to_save = []
        actions_to_save = []
        while not done:
            dqn_plays = observation["game_phase"] == GamePhase.BID
            playing_agent = bid_phase_dqn_agent if dqn_plays else random_agent
            action = playing_agent.get_action(observation)
            new_observation, rewards, done, _ = environment.step(action)
            if dqn_plays:
                observations_to_save.append(observation)
                actions_to_save.append(action)
            observation = new_observation
        rewards = rotate_list(rewards, environment.taking_player_original_id)
        for observation_to_save, action_to_save, reward in zip(observations_to_save, actions_to_save, rewards):
            reward_scaling_factor = 100.
            bid_phase_dqn_agent.memory.push(bid_phase_observation_encoder(observation_to_save).unsqueeze(0),
                                            action_to_save.value, None, reward / reward_scaling_factor)
        all_rewards.append(np.roll(rewards, i % environment.n_players))
        bid_phase_dqn_agent.optimize_model()
    return all_rewards


def _set_all_seeds(seed=1988):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_and_display_results(rewards, loss):
    rewards = np.stack(rewards)
    output_folder = "bid_agent_training_results"
    output_file_path = os.path.join(output_folder, "scores.csv")
    print("Dump scores at", output_file_path)
    columns = ["player_{}".format(i) for i in range(rewards.shape[1])]
    pd.DataFrame(rewards, columns=columns).to_csv(output_file_path)
    output_file_path = os.path.join(output_folder, "loss.csv")
    print("Dump loss at", output_file_path)

    print("average_loss_on_last_1000_episodes:", np.mean(loss[-1000:]))

    pd.DataFrame(loss, columns=["loss"]).to_csv(output_file_path)
    plt.subplot(211)
    plt.plot(rewards, alpha=0.5)
    plt.legend(columns)
    plt.xlabel("game")
    plt.ylabel("scores")
    plt.subplot(212)
    plt.plot(loss, label="loss")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training results")
    plt.show()


if __name__ == "__main__":
    _main()
