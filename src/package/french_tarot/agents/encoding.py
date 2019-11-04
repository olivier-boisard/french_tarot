from typing import List

import torch
from torch import tensor

from french_tarot.environment.core import Card, CARDS


def encode_cards(cards: List[Card]) -> torch.Tensor:
    return tensor([card in cards for card in CARDS]).float()
