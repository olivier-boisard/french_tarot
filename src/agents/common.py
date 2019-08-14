import random
from abc import abstractmethod, ABC
from collections import namedtuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import BCELoss

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


class TrainedPlayerNetwork(nn.Module):
    N_CARDS_PER_COLOR = 14
    N_TRUMPS_AND_EXCUSES = 22

    def __init__(self):
        super(TrainedPlayerNetwork, self).__init__()

        nn_width = 64
        self.standard_cards_tower = nn.Sequential(
            nn.Linear(TrainedPlayerNetwork.N_CARDS_PER_COLOR, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )
        self.trump_tower = nn.Sequential(
            nn.Linear(TrainedPlayerNetwork.N_TRUMPS_AND_EXCUSES, nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(nn_width),
            nn.Linear(nn_width, nn_width),
            nn.ReLU()
        )
        self.merge_tower = nn.Sequential(
            nn.BatchNorm1d(5 * nn_width),
            nn.Linear(5 * nn_width, 8 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, 8 * nn_width),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        n = TrainedPlayerNetwork.N_CARDS_PER_COLOR
        x_color_1 = self.standard_cards_tower(x[:, :n])
        x_color_2 = self.standard_cards_tower(x[:, n:2 * n])
        x_color_3 = self.standard_cards_tower(x[:, 2 * n:3 * n])
        x_color_4 = self.standard_cards_tower(x[:, 3 * n:4 * n])
        x_trumps = self.trump_tower(x[:, 4 * n:])
        x = torch.cat([x_color_1, x_color_2, x_color_3, x_color_4, x_trumps], dim=1)
        x = self.merge_tower(x)
        x = self.output_layer(x)
        return x


class Agent(ABC):
    def __init__(self, policy_net, eps_start=0.9, eps_end=0.05, eps_decay=500, batch_size=64,
                 replay_memory_size=2000):
        self._policy_net = policy_net
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
            reward_batch[reward_batch >= 0] = 1.
            reward_batch[reward_batch < 0.] = 0

            win_probability = self._policy_net(state_batch)
            loss = BCELoss()
            loss_output = loss(win_probability.flatten(), reward_batch.flatten())
            self.loss.append(loss_output.item())

            self._optimizer.zero_grad()
            loss_output.backward()
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.1)
            self._optimizer.step()

            if len(self.loss) % display_interval == 0:
                print("Loss:", np.mean(self.loss[-display_interval:]))

    @abstractmethod
    def get_action(self, observation):
        pass

    @staticmethod
    @abstractmethod
    def _create_qdn(self):
        pass

    @property
    def device(self):
        return "cuda" if next(self._policy_net.parameters()).is_cuda else "cpu"
