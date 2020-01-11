import json
from typing import Generator, List, IO

from french_tarot.reagent.data import ReAgentDataRow


class DataProducer:
    def __init__(self, data_generator: Generator[List[ReAgentDataRow], None, None], writer: IO):
        self._data_generator = data_generator
        self._writer = writer

    def run(self, n_max_episodes: int = None):
        running = True
        episode = 0
        while running:
            generated_data = next(self._data_generator)
            generated_data = [json.dumps(row.dictionary) + "\n" for row in generated_data]
            self._writer.writelines(generated_data)

            episode += 1
            running = episode < n_max_episodes
