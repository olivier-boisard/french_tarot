from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ModelUpdate:
    models: List[Dict]
