import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

from french_tarot.agents.common import set_all_seeds
from french_tarot.agents.performance_evaluation import compute_diff_score_metric
from french_tarot.agents.trained_player import AllPhasePlayerTrainer
from french_tarot.environment.core import Bid
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


def _main(n_episodes_training: int = 200000):
    set_all_seeds()
    writer = SummaryWriter()
    trained_agent = AllPhasePlayerTrainer(summary_writer=writer)
    _run_training(trained_agent, n_episodes_training, writer)


def _run_training(agent: AllPhasePlayerTrainer, n_episodes: int, tb_writer: SummaryWriter):
    environment = FrenchTarotEnvironment()
    for i in tqdm.tqdm(range(n_episodes)):
        observation = environment.reset()
        done = False

        early_phases_observations = []
        early_phases_actions = []
        rewards = None
        while not done:
            action = agent.get_action(observation)
            new_observation, rewards, done, _ = environment.step(action)

            if isinstance(observation, BidPhaseObservation) or isinstance(observation, DogPhaseObservation):
                early_phases_observations.append(observation)
                early_phases_actions.append(action)
            observation = new_observation

        if rewards is None:
            raise RuntimeError("No rewards set")
        # Add reward of creating the dog, i.e. append the reward that we got at the end of the game for the taker
        max_bid = np.max(environment._bid_per_player)
        if Bid.PASS < max_bid < Bid.GARDE_SANS:
            rewards.append(rewards[environment._taker_id])
        assert len(rewards) == len(early_phases_observations)
        assert len(rewards) == len(early_phases_actions)
        for observation, action, reward in zip(early_phases_observations, early_phases_actions, rewards):
            agent.push_to_agent_memory(observation, action, reward)

        agent.optimize_model()

        if i % 1000 == 0:
            tb_writer.add_scalar("score_diff", compute_diff_score_metric(agent), i)


if __name__ == "__main__":
    _main()
