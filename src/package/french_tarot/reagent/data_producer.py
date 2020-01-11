from typing import Generator, List

from french_tarot.reagent.data import ReAgentDataRow
from french_tarot.reagent.saving.data_saver import BaseDataSaver


class DataProducer:
    def __init__(self, data_generator: Generator[List[ReAgentDataRow], None, None], data_saver: BaseDataSaver):
        self._data_generator = data_generator
        self._data_saver = data_saver

    def run(self, n_max_episodes: int = None):
        running = True
        episode = 0
        while running:
            generated_data = next(self._data_generator)
            self._data_saver.save_list(generated_data)

            episode += 1
            running = episode < n_max_episodes
