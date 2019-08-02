import copy
import datetime
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from joblib import Parallel, delayed

from agent import RandomPlayer
from environment import FrenchTarotEnvironment


def _main():
    scores = Parallel(n_jobs=-1, verbose=10)(delayed(_run_game)(i) for i in tqdm.tqdm(range(100)))

    _print_final_scores(scores)
    _plot_scores(scores)


def _run_game(iteration):
    environment = FrenchTarotEnvironment(seed=iteration)
    random_agent = RandomPlayer(seed=iteration)
    observation = environment.reset()
    done = False
    cnt = 0
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


def _print_final_scores(scores):
    final_scores = np.stack(scores)
    if np.sum(final_scores) != 0:
        RuntimeError("Scores do not sum up to 0")
    print("Final scores: ", final_scores.sum(axis=0))


def _plot_scores(scores):
    plt.plot(np.stack(scores).cumsum(axis=0))
    plt.xlabel("Game")
    plt.ylabel("Scores")
    plt.legend(["player " + str(player_id) for player_id in range(len(scores[0]))])
    plt.title("Score per player evolution")
    plt.show()


if __name__ == "__main__":
    _main()
