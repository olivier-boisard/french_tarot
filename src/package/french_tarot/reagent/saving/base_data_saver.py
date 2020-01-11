from abc import abstractmethod
from typing import List

from french_tarot.reagent.data import ReAgentDataRow


class BaseDataSaver:

    @abstractmethod
    def save_list(self, rows: List[ReAgentDataRow]):
        pass
