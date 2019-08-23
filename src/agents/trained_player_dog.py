import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

from agents.common import Agent, card_set_encoder, Transition
from environment import Card, GamePhase


def _card_is_ok_in_dog(card):
    return "trump" not in card.value and "king" not in card.value and "excuse" not in card.value


class DogPhaseAgent(Agent):
    """
    Somewhat inspired from this: https://arxiv.org/pdf/1711.08946.pdf
    """
    CARDS_OK_IN_DOG = [card for card in list(Card) if _card_is_ok_in_dog(card)]
    OUTPUT_DIMENSION = len(CARDS_OK_IN_DOG)

    def __init__(self, device="cuda", **kwargs):
        super(DogPhaseAgent, self).__init__(DogPhaseAgent._create_dqn().to(device), **kwargs)

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.DOG:
            raise ValueError("Game is not in dog phase")

        hand = list(observation["hand"])
        selected_cards = torch.zeros(DogPhaseAgent.OUTPUT_DIMENSION)
        dog_size = len(observation["original_dog"])
        for _ in range(dog_size):
            xx = torch.cat([card_set_encoder(hand), selected_cards])
            xx = self._policy_net(xx.to(self.device))
            mask = [card not in hand for card in DogPhaseAgent.CARDS_OK_IN_DOG]
            xx[mask] = 0
            selected_card_index = xx.argmax()
            selected_cards[selected_card_index] = 1
            hand.remove(DogPhaseAgent.CARDS_OK_IN_DOG[selected_card_index])
        assert selected_cards.sum() == dog_size
        return list(np.array(DogPhaseAgent.CARDS_OK_IN_DOG)[np.array(selected_cards, dtype=bool)])

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
        return nn.Sequential(nn.Linear(78 + DogPhaseAgent.OUTPUT_DIMENSION, DogPhaseAgent.OUTPUT_DIMENSION))

    def optimize_model(self):
        """
        See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        """
        display_interval = 100
        return_scale_factor = 0.001
        if len(self.memory) > self._batch_size:
            transitions = self.memory.sample(self._batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
            return_batch = torch.tensor(batch.reward).float().to(self.device) * return_scale_factor

            estimated_return = self._policy_net(state_batch).gather(1, action_batch)

            loss_output = smooth_l1_loss(estimated_return, return_batch)
            self.loss.append(loss_output.item())

            self._optimizer.zero_grad()
            loss_output.backward()
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.1)
            self._optimizer.step()

            if len(self.loss) % display_interval == 0:
                print("Loss for dog agent:", np.mean(self.loss[-display_interval:]))
