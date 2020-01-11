from typing import List

from french_tarot.reagent.data import ReAgentDataRow
from french_tarot.reagent.data_producer import DataProducer
from french_tarot.reagent.saving.data_saver import BaseDataSaver


def test_data_producer(mocker):
    data_generator = fake_generator(mocker.Mock())

    data_saver = FakeDataSaver()
    producer = DataProducer(data_generator, data_saver)
    producer.run(n_max_episodes=1)
    assert data_saver.rows is not None


def fake_generator(dummy_data):
    while True:
        yield dummy_data


class FakeDataSaver(BaseDataSaver):

    def __init__(self):
        self.rows = None

    def save_list(self, rows: List[ReAgentDataRow]):
        self.rows = rows
