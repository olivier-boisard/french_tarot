from typing import Dict

import torch
from attr import dataclass

from french_tarot.agents.common import BaseNeuralNetAgent


@dataclass
class ModelUpdate:
    agent_to_model_map: Dict[BaseNeuralNetAgent, torch.nn.Module]
