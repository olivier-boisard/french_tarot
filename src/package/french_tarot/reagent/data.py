from _ast import Dict
from dataclasses import dataclass
from typing import List, Union


@dataclass
class ReAgentDataRow:
    mdp_id: int
    sequence_number: str
    state_features: Dict
    action: int
    reward: float
    possible_actions: List[int]
    action_probability: Union[int, None]
    ds: str

    @property
    def dictionary(self):
        self_as_dict = vars(self)
        self_as_dict["mdp_id"] = str(self_as_dict["mdp_id"])
        self_as_dict["state_features"] = {str(key): str(value) for key, value in self_as_dict["state_features"].items()}
        self_as_dict["action"] = self_as_dict["action"]
        self_as_dict["possible_actions"] = list(map(str, self_as_dict["possible_actions"]))
        return self_as_dict
