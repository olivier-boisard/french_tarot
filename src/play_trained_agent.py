from trained_player import BidPhaseAgent


def _main():
    # Mostly got inspiration from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    policy_net = BidPhaseAgent.create_dqn()
    target_net = BidPhaseAgent.create_dqn()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    BidPhaseAgent(policy_net)
    

if __name__ == "__main__":
    _main()
