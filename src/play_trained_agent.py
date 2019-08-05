import torch
import tqdm

from environment import FrenchTarotEnvironment, GamePhase
from random_agent import RandomPlayer
from trained_player import BidPhaseAgent, bid_phase_observation_encoder


def _main():
    torch.manual_seed(1988)
    torch.cuda.manual_seed_all(1988)

    # Mostly got inspiration from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    policy_net = BidPhaseAgent.create_dqn()

    environment = FrenchTarotEnvironment()
    bid_phase_dqn_agent = BidPhaseAgent(policy_net)
    random_agent = RandomPlayer()
    for _ in tqdm.tqdm(range(1000)):
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
                                            action_to_save.value, None, reward / 1000.)
        bid_phase_dqn_agent.optimize_model()


if __name__ == "__main__":
    _main()
