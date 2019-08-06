import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from environment import FrenchTarotEnvironment, GamePhase
from random_agent import RandomPlayer
from trained_player import BidPhaseAgent, bid_phase_observation_encoder


def _main():
    seed = 1988
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Mostly got inspiration from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    policy_net = BidPhaseAgent.create_dqn()

    environment = FrenchTarotEnvironment()
    bid_phase_dqn_agent = BidPhaseAgent(policy_net)
    random_agent = RandomPlayer()
    all_rewards = []
    for i in tqdm.tqdm(range(200000)):
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
        for observation_to_save, action_to_save, reward in zip(observations_to_save, actions_to_save, rewards):
            bid_phase_dqn_agent.memory.push(bid_phase_observation_encoder(observation_to_save).unsqueeze(0),
                                            action_to_save.value, None, reward / 100.)
        all_rewards.append(np.roll(rewards, i % environment.n_players))
        bid_phase_dqn_agent.optimize_model()

    dump_and_display_results(all_rewards, bid_phase_dqn_agent.loss)


def dump_and_display_results(rewards, loss):
    rewards = np.stack(rewards)
    output_file_path = "scores.csv"
    print("Dump scores at", output_file_path)
    columns = ["player_{}".format(i) for i in range(rewards.shape[1])]
    pd.DataFrame(rewards, columns=columns).to_csv(output_file_path)
    output_file_path = "loss.csv"
    print("Dump loss at", output_file_path)
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
