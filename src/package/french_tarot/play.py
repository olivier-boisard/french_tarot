import torch
import tqdm as tqdm

from french_tarot.agents.trained_player import AllPhaseAgent
from french_tarot.environment.french_tarot import FrenchTarotEnvironment


def main():
    n_episodes_training: int = 200000

    _set_all_seeds()
    _run_episodes(AllPhaseAgent(), n_episodes_training)


def _run_episodes(agent: AllPhaseAgent, n_episodes: int):
    environment = FrenchTarotEnvironment()
    for _ in tqdm.tqdm(range(n_episodes)):
        observation = environment.reset()
        done = False

        while not done:
            action = agent.get_action(observation)
            observation, rewards, done, _ = environment.step(action)


def _set_all_seeds(seed: int = 0):
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
