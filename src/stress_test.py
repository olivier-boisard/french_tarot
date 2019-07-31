import copy
import os

import dill
import tqdm

from agent import RandomPlayer
from environment import FrenchTarotEnvironment


def _main():
    environment = FrenchTarotEnvironment()
    random_agent = RandomPlayer()
    for i in tqdm.tqdm(range(1000)):
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
                output_file_path = os.path.join(os.path.dirname(__file__), "stress_test_{}.dill".format(i))
                print("Dumping file into " + output_file_path)
                with open(output_file_path, "wb") as f:
                    dill.dump(obj, f)
                raise e
            cnt += 1
            if cnt >= 1000:
                raise RuntimeError("Infinite loop")


if __name__ == "__main__":
    _main()
