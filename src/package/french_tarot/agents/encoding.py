from typing import List

import torch
from torch import tensor

from french_tarot.environment.core import Card, CARDS


def encode_cards(cards):
    return [card in cards for card in CARDS]


def encode_cards_as_tensor(cards: List[Card]) -> torch.Tensor:
    return tensor(encode_cards(cards)).float()
