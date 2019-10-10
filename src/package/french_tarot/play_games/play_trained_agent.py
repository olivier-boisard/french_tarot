import tqdm

from french_tarot.agents.common import set_all_seeds
from french_tarot.agents.trained_player import AllPhasePlayerTrainer
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.bid_phase import BidPhaseObservation
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


def _main(n_episodes_training: int = 200000):
    set_all_seeds()
    trained_agent = AllPhasePlayerTrainer()
    _run_training(trained_agent, n_episodes_training)


def _run_training(agent: AllPhasePlayerTrainer, n_episodes: int):
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

        assert rewards is not None
        dog_reward = environment.extract_dog_phase_reward(rewards)
        if dog_reward is not None:
            rewards.append(dog_reward)
        assert len(rewards) == len(early_phases_observations)
        assert len(rewards) == len(early_phases_actions)
        for observation, action, reward in zip(early_phases_observations, early_phases_actions, rewards):
            agent.push_to_agent_memory(observation, action, reward)

        agent.optimize_model()


if __name__ == "__main__":
    _main()
