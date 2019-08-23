import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import BCELoss

from agents.common import Agent, card_set_encoder, Transition
from environment import Card, GamePhase


def _card_is_ok_in_dog(card):
    return "trump" not in card.value and "king" not in card.value and "excuse" not in card.value


class DogPhaseAgent(Agent):
    """
    Somewhat inspired from this: https://arxiv.org/pdf/1711.08946.pdf
    """
    CARDS_OK_IN_DOG = [card for card in list(Card) if _card_is_ok_in_dog(card)]

    def __init__(self, device="cuda", **kwargs):
        super(DogPhaseAgent, self).__init__(DogPhaseAgent._create_dqn().to(device), **kwargs)

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.DOG:
            raise ValueError("Game is not in dog phase")

        xx = card_set_encoder(observation)
        xx = self._policy_net(xx.to(self.device))
        return DogPhaseAgent._select_cards(
            xx,
            np.concatenate((observation["hand"], observation["original_dog"])),
            len(observation["original_dog"])
        )

    @staticmethod
    def _select_cards(xx, hand, n_cards):
        xx = xx.clone().detach()
        assert xx.size(0) == len(DogPhaseAgent.CARDS_OK_IN_DOG)
        mask = [card not in hand for card in DogPhaseAgent.CARDS_OK_IN_DOG]
        xx[mask] = 0

        indices = DogPhaseAgent._cards_selection_mask(xx, n_cards)
        new_dog = list(np.array(DogPhaseAgent.CARDS_OK_IN_DOG)[np.array(indices.cpu().numpy(), dtype=np.bool)])
        return new_dog

    @staticmethod
    def _cards_selection_mask(model_output, n_cards):
        model_output = model_output.clone().detach()
        selections = torch.zeros_like(model_output)
        for _ in range(n_cards):
            i = model_output.argmax()
            model_output[i] = 0
            selections[i] = 1
        assert selections.sum().item() == n_cards
        return selections

    @staticmethod
    def _create_dqn():
        return nn.Sequential(nn.Linear(78, 52), nn.Sigmoid())

    def optimize_model(self):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        display_interval = 100
        if len(self.memory) > self._batch_size:
            transitions = self.memory.sample(self._batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            return_batch = torch.tensor(batch.reward).float().to(self.device)

            estimated_return = self._policy_net(state_batch)
            loss = BCELoss()

            loss_output = loss(estimated_return, return_batch)
            self.loss.append(loss_output.item())

            self._optimizer.zero_grad()
            loss_output.backward()
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.1)
            self._optimizer.step()

            if len(self.loss) % display_interval == 0:
                print("Loss:", np.mean(self.loss[-display_interval:]))
