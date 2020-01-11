from typing import List

from french_tarot.reagent.data import ReAgentDataRow
from french_tarot.reagent.data_producer import DataProducer


def test_data_producer(mocker):
    mocked_data_row = mocker.Mock()
    mocked_data_row.dictionary = {"a": "b"}
    data_generator = fake_generator(mocked_data_row)

    data_saver = FakeDataSaver()
    producer = DataProducer(data_generator, data_saver)
    producer.run(n_max_episodes=1)
    assert data_saver.rows is not None


def fake_generator(dummy_data):
    while True:
        yield [dummy_data]


class FakeDataSaver:

    def __init__(self):
        self.rows = None

    def writelines(self, rows: List[ReAgentDataRow]):
        self.rows = rows
