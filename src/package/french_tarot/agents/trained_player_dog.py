import copy
import itertools
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

from french_tarot.agents.common import BaseNeuralNetAgent, encode_cards, CoreCardNeuralNet, Trainer, Transition
from french_tarot.environment.core import Card, CARDS
from french_tarot.environment.subenvironments.dog_phase import DogPhaseObservation


def _card_is_ok_in_dog(card: Card) -> bool:
    return "trump" not in card.value and "king" not in card.value and "excuse" not in card.value


class DogPhaseAgent(BaseNeuralNetAgent):
    """
    Somewhat inspired from this: https://arxiv.org/pdf/1711.08946.pdf
    """

    CARDS_OK_IN_DOG = [card for card in CARDS if _card_is_ok_in_dog(card)]
    CARDS_OK_IN_DOG_WITH_TRUMPS = [card for card in CARDS if _card_is_ok_in_dog(card) or "trump" in card.value]

    def __init__(self, policy_net: torch.nn.Module, seed: int = 1988):
        super().__init__(policy_net)
        self._random_state = np.random.RandomState(seed)

    def get_max_return_action(self, observation: DogPhaseObservation):
        hand = copy.copy(observation.player.hand)
        selected_cards = torch.zeros(len(CARDS))
        for _ in range(observation.dog_size):
            xx = torch.cat([encode_cards(hand), selected_cards]).unsqueeze(0)
            xx = self.policy_net(xx.to(self.device)).squeeze()

            xx[~DogPhaseAgent._get_card_selection_mask(hand)] = -np.inf
            selected_card_index = xx.argmax()
            selected_cards[selected_card_index] = 1
            hand.remove(CARDS[selected_card_index])
        assert selected_cards.sum() == observation.dog_size
        return list(np.array(CARDS)[np.array(selected_cards, dtype=bool)])

    def get_random_action(self, observation: DogPhaseObservation):
        cards_ok_in_dog = self._get_card_selection_mask(observation.player.hand)
        index = self._random_state.choice(np.where(cards_ok_in_dog)[0], size=observation.dog_size, replace=False)
        return list(np.array(CARDS)[index])

    @staticmethod
    def _get_card_selection_mask(hand: List[Card]) -> np.ndarray:
        mask = [card in hand and card in DogPhaseAgent.CARDS_OK_IN_DOG for card in Card]
        if len(mask) == np.sum(mask):
            mask = [card not in hand or card not in DogPhaseAgent.CARDS_OK_IN_DOG_WITH_TRUMPS for card in Card]
        assert np.sum(mask) > 0
        return np.array(mask)

    @staticmethod
    def _cards_selection_mask(model_output: torch.Tensor, n_cards: int) -> torch.Tensor:
        model_output = model_output.clone().detach()
        selections = torch.zeros_like(model_output)
        for _ in range(n_cards):
            i = model_output.argmax()
            model_output[i] = 0
            selections[i] = 1
        assert selections.sum().item() == n_cards
        return selections

    @staticmethod
    def create_dqn(base_neural_net: nn.Module) -> nn.Module:
        return TrainedPlayerDogNeuralNet(base_neural_net)


class DogPhaseAgentTrainer(Trainer):

    def __init__(self, net: torch.nn.Module, **kwargs):
        # noinspection PyArgumentList
        super().__init__(net, **kwargs)
        self._return_scale_factor = 0.001

    def get_model_output_and_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state_batch, action_batch, target = self._get_batches()
        model_output = self.model(state_batch).gather(1, action_batch)
        return model_output, target

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = smooth_l1_loss(model_output.squeeze(), target)
        return loss

    def push_to_memory(self, observation: DogPhaseObservation, action, reward):
        selected_cards = torch.zeros(len(CARDS))
        for permuted_action in itertools.permutations(action):
            hand = list(observation.player.hand)
            for card in permuted_action:
                xx = torch.cat((encode_cards(hand), selected_cards)).unsqueeze(0)
                action_id = DogPhaseAgent.CARDS_OK_IN_DOG.index(card)
                self._memory.push_message(xx, action_id, None, reward)

                selected_cards[action_id] = 1
                hand.remove(DogPhaseAgent.CARDS_OK_IN_DOG[action_id])

    def _get_batches(self):
        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        return_batch = torch.tensor(batch.reward).float().to(self.device) * self._return_scale_factor
        return state_batch, action_batch, return_batch


class TrainedPlayerDogNeuralNet(nn.Module):

    def __init__(self, base_card_neural_net: CoreCardNeuralNet):
        super().__init__()
        self.base_card_neural_net = base_card_neural_net
        nn_width = base_card_neural_net.output_dimensions
        self.merge_tower = nn.Sequential(
            nn.BatchNorm1d(2 * nn_width),
            nn.Linear(2 * nn_width, 4 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(4 * nn_width),
            nn.Linear(4 * nn_width, 4 * nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(4 * nn_width),
            nn.Linear(4 * nn_width, 8 * nn_width),
            nn.ReLU(),
            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, 8 * nn_width),
            nn.ReLU(),

            nn.BatchNorm1d(8 * nn_width),
            nn.Linear(8 * nn_width, len(CARDS))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hand_end_idx = x.size(1) // 2
        x_hand = self.base_card_neural_net(x[:, :hand_end_idx])
        x_feedback = self.base_card_neural_net(x[:, hand_end_idx:])
        return self.merge_tower(torch.cat([x_hand, x_feedback], dim=1))
