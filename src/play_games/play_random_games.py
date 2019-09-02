import argparse
import copy
import datetime
import os
from typing import List

import dill
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from agents.random_agent import RandomPlayer
from environment import FrenchTarotEnvironment

N_ITERATIONS = 1000


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-seed", default=0, type=int)
    args = parser.parse_args()

    scores = Parallel(n_jobs=-1, verbose=1)(
        delayed(_run_game)(i, initial_seed=args.initial_seed) for i in range(N_ITERATIONS)
    )

    _print_final_scores(scores)
    _plot_scores(scores)


def _run_game(iteration: int, initial_seed: int = 0) -> np.array:
    environment = FrenchTarotEnvironment(seed=initial_seed + iteration)
    random_agent = RandomPlayer(seed=initial_seed + iteration)
    observation = environment.reset()
    done = False
    cnt = 0
    reward = None
    while not done:
        environment_copy = copy.deepcopy(environment)
        random_agent_copy = copy.deepcopy(random_agent)
        try:
            observation, reward, done, _ = environment.step(random_agent.get_action(observation))
        except ValueError as e:
            obj = {"agent": random_agent_copy, "environment": environment_copy, "observation": observation,
                   "done": done}
            timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "-")
            output_file_path = os.path.join(os.path.dirname(__file__), "stress_test_{}.dill".format(timestamp))
            print("Dumping file into " + output_file_path)
            with open(output_file_path, "wb") as f:
                dill.dump(obj, f)
            raise e
        cnt += 1
        if cnt >= 1000:
            raise RuntimeError("Infinite loop")
    offset = iteration % environment.n_players  # first player changes at each turn
    game_scores = np.roll(np.array(reward)[observation["original_player_ids"]], offset)
    if np.sum(game_scores) != 0:
        RuntimeError("Scores do not sum up to 0")
    return game_scores


def _print_final_scores(scores: List[np.array]):
    final_scores = np.stack(scores)
    if np.sum(final_scores) != 0:
        RuntimeError("Scores do not sum up to 0")
    print("Final scores: ", final_scores.sum(axis=0))


def _plot_scores(scores: List[np.array]):
    plt.plot(np.stack(scores).cumsum(axis=0))
    plt.xlabel("Game")
    plt.ylabel("Scores")
    plt.legend(["player " + str(player_id) for player_id in range(len(scores[0]))])
    plt.title("Score per player with random agent")
    plt.show()


if __name__ == "__main__":
    _main()
