from french_tarot.agents.common import set_all_seeds
from french_tarot.agents.trained_player import AllPhasePlayerTrainer


def _main(n_episodes_training: int = 200000):
    set_all_seeds()
    trained_agent = AllPhasePlayerTrainer()
    _run_training(trained_agent, n_episodes_training)


def _run_training(agent: AllPhasePlayerTrainer, n_episodes: int):
    raise NotImplementedError


if __name__ == "__main__":
    _main()
