from torch import optim

from trained_player import BidPhaseAgent, ReplayMemory


def _main():
    # Mostly got inspiration from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    policy_net = BidPhaseAgent.create_dqn()
    target_net = BidPhaseAgent.create_dqn()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)


if __name__ == "__main__":
    _main()
