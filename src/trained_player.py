import math
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, tensor, optim

from environment import Card, Bid, GamePhase


def bid_phase_observation_encoder(observation):
    # TODO this is called twice which is not optimal
    return tensor([card in observation["hand"] for card in list(Card)]).float()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Got from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self._random_state = random.Random(1988)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self._random_state.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BidPhaseAgent:

    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=5000, batch_size=128,
                 replay_memory_size=20000, device="cuda"):
        self._policy_net = BidPhaseAgent._create_dqn().to(device)
        self._steps_done = 0
        self._random_state = np.random.RandomState(1988)

        # Training parameters
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._batch_size = batch_size
        self.memory = ReplayMemory(replay_memory_size)
        self._optimizer = optim.Adam(self._policy_net.parameters())

        self.loss = []

    @property
    def device(self):
        return "cuda" if next(self._policy_net.parameters()).is_cuda else "cpu"

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.BID:
            raise ValueError("Invalid game phase")

        state = bid_phase_observation_encoder(observation)

        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(
            -1. * self._steps_done / self._eps_decay)
        self._steps_done += 1
        if self._random_state.rand() > eps_threshold:
            with torch.no_grad():
                self._policy_net.eval()
                output = self._policy_net(state.unsqueeze(0).to(self.device)).argmax().item()
                self._policy_net.train()
        else:
            output = torch.argmax(torch.tensor([self._random_state.rand(self.output_dimension)])).item()

        if len(observation["bid_per_player"]) > 0:
            if np.max(observation["bid_per_player"]) >= output:
                output = Bid.PASS
        return Bid(output)

    @property
    def output_dimension(self):
        return self._policy_net[-1].out_features

    @staticmethod
    def _create_dqn():
        input_size = len(list(Card))
        nn_width = 128
        return nn.Sequential(
            nn.Linear(input_size, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, 2 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(2 * nn_width),
            nn.Linear(2 * nn_width, 2 * nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(2 * nn_width),
            nn.Linear(2 * nn_width, len(list(Bid)))
        )

    def optimize_model(self):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        display_interval = 100
        if len(self.memory) > self._batch_size:
            transitions = self.memory.sample(self._batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            reward_batch = torch.tensor(batch.reward).float().to(self.device)
            action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)

            state_action_values = self._policy_net(state_batch).gather(1, action_batch)
            loss = F.smooth_l1_loss(state_action_values, reward_batch.unsqueeze(1))
            self.loss.append(loss.item())

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if len(self.loss) % display_interval == 0:
                print("Loss:", np.mean(self.loss[-display_interval:]))


class TrainedPlayerNetwork(nn.Module):
    N_CARDS_PER_COLOR = 14
    N_TRUMPS = 21

    def __init__(self):
        nn_width = 64
        self.standard_cards_tower = nn.Sequential(
            nn.Linear(TrainedPlayerNetwork.N_CARDS_PER_COLOR, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )
        self.trump_tower = nn.Sequential(
            nn.Linear(TrainedPlayerNetwork.N_TRUMPS, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )

    def forward(self, x):
        n = TrainedPlayerNetwork.N_CARDS_PER_COLOR
        x_color_1 = self.standard_cards_tower(x[:, n])
        x_color_2 = self.standard_cards_tower(x[:, n:2 * n])
        x_color_3 = self.standard_cards_tower(x[:, 2 * n:3 * n])
        x_color_4 = self.standard_cards_tower(x[:, 3 * n:4 * n])
        x_trumps = self.trump_tower(x[:, 4 * n:])
        x_concat = nn.cat()
