import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from torch.utils.tensorboard import SummaryWriter

from agents.common import card_set_encoder, set_all_seeds, BaseCardNeuralNet
from agents.random_agent import RandomPlayer
from agents.trained_player_bid import BidPhaseAgent
from environment import FrenchTarotEnvironment, GamePhase, rotate_list


def _main():
    set_all_seeds()
    bid_phase_dqn_agent = BidPhaseAgent(BaseCardNeuralNet())
    all_rewards_training = _run_training(bid_phase_dqn_agent)
    dump_and_display_results(all_rewards_training, bid_phase_dqn_agent.loss)


def _run_training(bid_phase_dqn_agent: BidPhaseAgent, n_iterations: int = 20000) -> List[np.array]:
    environment = FrenchTarotEnvironment()
    random_agent = RandomPlayer()
    all_rewards = []
    tb_writer = SummaryWriter()
    for i in tqdm.tqdm(range(n_iterations)):
        observation = environment.reset()
        done = False
        observations_to_save = []
        actions_to_save = []
        rewards = None
        while not done:
            dqn_plays = observation["game_phase"] == GamePhase.BID
            playing_agent = bid_phase_dqn_agent if dqn_plays else random_agent
            action = playing_agent.get_action(observation)
            new_observation, rewards, done, _ = environment.step(action)
            if dqn_plays:
                observations_to_save.append(observation)
                actions_to_save.append(action)
            observation = new_observation
        original_id = environment.taking_player_original_id
        rewards = rotate_list(rewards, original_id)
        bid_phase_dqn_agent.memory.push(
            card_set_encoder(observations_to_save[original_id]["hand"]).unsqueeze(0),
            None, None, rewards[original_id]
        )
        all_rewards.append(np.roll(rewards, i % environment.n_players))
        bid_phase_dqn_agent.optimize_model(tb_writer)
    return all_rewards


def dump_and_display_results(rewards: List[np.array], loss: np.array):
    rewards = np.stack(rewards)
    output_folder = "bid_agent_training_results"
    if not os.path.isdir(output_folder):
        print("Creating directory at", output_folder)
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, "scores.csv")
    print("Dump scores at", output_file_path)
    columns = ["player_{}".format(i) for i in range(rewards.shape[1])]
    rewards = pd.DataFrame(rewards, columns=columns)
    rewards.to_csv(output_file_path)
    print("% of the last 1000 games where somebody took the game:", (rewards.iloc[-1000:] != 0).mean() * 100)
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
