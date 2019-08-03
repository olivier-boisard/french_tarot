import numpy as np
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
    target_net = BidPhaseAgent.create_dqn()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    environment = FrenchTarotEnvironment()
    bid_phase_dqn_agent = BidPhaseAgent(policy_net)
    random_agent = RandomPlayer()
    random_state = np.random.RandomState(1988)
    for _ in tqdm.tqdm(range(1000)):
        observation = environment.reset()
        trained_agent_id = random_state.randint(4)
        done = False
        observation_to_save = None
        action_to_save = None
        while not done:
            dqn_plays = environment.current_player == trained_agent_id and observation["game_phase"] == GamePhase.BID
            playing_agent = bid_phase_dqn_agent if dqn_plays else random_agent
            action = playing_agent.get_action(observation)
            new_observation, reward, done, _ = environment.step(action)
            if dqn_plays:
                observation_to_save = observation
                action_to_save = action
            observation = new_observation
        bid_phase_dqn_agent.memory.push(bid_phase_observation_encoder(observation_to_save), action_to_save.value, None,
                                        reward[trained_agent_id])
        bid_phase_dqn_agent.optimize_model()


if __name__ == "__main__":
    _main()
