import copy
from typing import List

import numpy as np
import torch

from french_tarot.agents.encoding import encode_cards
from french_tarot.agents.neural_net import BaseNeuralNetAgent
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
