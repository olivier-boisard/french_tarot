from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter

from agents.common import BaseNeuralNetAgent, card_set_encoder, Transition, BaseCardNeuralNet
from environment import Card, GamePhase


def _card_is_ok_in_dog(card: Card) -> bool:
    return "trump" not in card.value and "king" not in card.value and "excuse" not in card.value


class DogPhaseAgent(BaseNeuralNetAgent):
    """
    Somewhat inspired from this: https://arxiv.org/pdf/1711.08946.pdf
    """
    # noinspection PyTypeChecker
    CARDS_OK_IN_DOG = [card for card in list(Card) if _card_is_ok_in_dog(card)]
    # noinspection PyTypeChecker
    CARDS_OK_IN_DOG_WITH_TRUMPS = [card for card in list(Card) if _card_is_ok_in_dog(card) or "trump" in card.value]

    def __init__(self, base_card_neural_net, device: str = "cuda", **kwargs):
        # noinspection PyUnresolvedReferences
        super(DogPhaseAgent, self).__init__(DogPhaseAgent._create_dqn(base_card_neural_net).to(device), **kwargs)
        self._epoch = 0

    def get_action(self, observation: dict):
        if observation["game_phase"] != GamePhase.DOG:
            raise ValueError("Game is not in dog phase")

        hand = list(observation["hand"])
        # noinspection PyTypeChecker
        selected_cards = torch.zeros(len(list(Card)))
        dog_size = len(observation["original_dog"])
        for _ in range(dog_size):
            xx = torch.cat([card_set_encoder(hand), selected_cards]).unsqueeze(0)
            self._policy_net.eval()
            xx = self._policy_net(xx.to(self.device)).squeeze()
            self._policy_net.train()

            xx[DogPhaseAgent._get_card_selection_mask(hand)] = -np.inf
            selected_card_index = xx.argmax()
            selected_cards[selected_card_index] = 1
            # noinspection PyTypeChecker
            hand.remove(list(Card)[selected_card_index])
        assert selected_cards.sum() == dog_size
        # noinspection PyTypeChecker
        return list(np.array(Card)[np.array(selected_cards, dtype=bool)])

    @staticmethod
    def _get_card_selection_mask(hand: List[Card]) -> List[bool]:
        mask = [card not in hand or card not in DogPhaseAgent.CARDS_OK_IN_DOG for card in Card]
        if len(mask) == np.sum(mask):
            mask = [card not in hand or card not in DogPhaseAgent.CARDS_OK_IN_DOG_WITH_TRUMPS for card in Card]
        assert np.sum(mask) > 0
        return mask

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
    def _create_dqn(base_neural_net: nn.Module) -> nn.Module:
        return TrainedPlayerDogNeuralNet(base_neural_net)

    def optimize_model(self, tb_writer: SummaryWriter):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        return_scale_factor = 0.001
        if len(self.memory) > self._batch_size:
            transitions = self.memory.sample(self._batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
            return_batch = torch.tensor(batch.reward).float().to(self.device) * return_scale_factor

            estimated_return = self._policy_net(state_batch).gather(1, action_batch)

            loss_output = smooth_l1_loss(estimated_return.squeeze(), return_batch)
            self.loss.append(loss_output.item())

            self._optimizer.zero_grad()
            loss_output.backward()
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.1)
            self._optimizer.step()

            if self._epoch % 1000 == 0:
                tb_writer.add_scalar("Loss/train/Dog", loss_output.item(), self._epoch)
            self._epoch += 1


class TrainedPlayerDogNeuralNet(nn.Module):

    def __init__(self, base_card_neural_net: BaseCardNeuralNet):
        super(TrainedPlayerDogNeuralNet, self).__init__()
        self.base_card_neural_net = base_card_neural_net
        nn_width = base_card_neural_net.output_dimensions
        # noinspection PyTypeChecker
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
            nn.Linear(8 * nn_width, len(list(Card)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hand_end_idx = x.size(1) // 2
        x_hand = self.base_card_neural_net(x[:, :hand_end_idx])
        x_feedback = self.base_card_neural_net(x[:, hand_end_idx:])
        return self.merge_tower(torch.cat([x_hand, x_feedback], dim=1))
