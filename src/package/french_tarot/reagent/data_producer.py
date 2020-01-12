import json
from typing import Generator, List, IO

from french_tarot.reagent.data import ReAgentDataRow


class DataProducer:
    def __init__(self, data_generator: Generator[List[ReAgentDataRow], None, None], writer: IO):
        self._data_generator = data_generator
        self._writer = writer
        self._running = False

    def run(self, n_max_episodes: int = None):
        episode = 0
        self._running = True
        while self._running:
            generated_data = next(self._data_generator)
            generated_data = [json.dumps(row.dictionary) + "\n" for row in generated_data]
            self._writer.writelines(generated_data)

            episode += 1
            if n_max_episodes is not None:
                self._running = episode < n_max_episodes

    def stop(self):
        self._running = False
