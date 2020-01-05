import ast
import os
import shutil

import pytest

from french_tarot.play import play_episodes
from french_tarot.reagent.wrapper import convert_to_timeline_format, _generate_timeline, _get_reagent_directory


@pytest.fixture
def batch():
    n_episodes = 10
    return play_episodes(n_episodes)


def test_convert_to_timeline_format(request, batch):
    output_folder = "dummy_training_data"
    base_name = "french_tarot_discrete"
    tmp_training_data_dir = os.path.join(_get_reagent_directory(), "%s_training" % base_name)
    tmp_eval_data_dir = os.path.join(_get_reagent_directory(), "%s_eval" % base_name)

    request.addfinalizer(lambda: shutil.rmtree(output_folder))
    request.addfinalizer(lambda: os.remove(os.path.join(_get_reagent_directory(), base_name)))
    request.addfinalizer(lambda: shutil.rmtree(tmp_training_data_dir, ignore_errors=True))
    request.addfinalizer(lambda: shutil.rmtree(tmp_eval_data_dir, ignore_errors=True))

    convert_to_timeline_format(batch, output_folder)
    assert os.path.isfile(os.path.join(output_folder, "french_tarot_discrete_timeline_training.json"))
    assert os.path.isfile(os.path.join(output_folder, "french_tarot_discrete_timeline_eval.json"))
    assert not os.path.isdir(tmp_training_data_dir)
    assert not os.path.isdir(tmp_eval_data_dir)


def test_generate_timeline(batch):
    output = _generate_timeline(batch, "dummy_table_name", table_sample=5)
    timeline = output.timeline
    query = output.query
    assert isinstance(timeline["startDs"], str)
    assert isinstance(timeline["endDs"], str)
    assert isinstance(timeline["addTerminalStateRow"], bool)
    assert isinstance(timeline["actionDiscrete"], bool)
    assert isinstance(timeline["inputTableName"], str)
    assert isinstance(timeline["outputTableName"], str)
    assert isinstance(timeline["evalTableName"], str)
    assert timeline["inputTableName"] != timeline["outputTableName"]
    assert timeline["inputTableName"] != timeline["evalTableName"]
    assert timeline["outputTableName"] != timeline["evalTableName"]
    assert isinstance(timeline["numOutputShards"], int)
    assert isinstance(query["tableSample"], int)
    assert isinstance(query["actions"], list)
    assert all([isinstance(action, str) for action in query["actions"]])
    assert all([isinstance(ast.literal_eval(action), int) for action in query["actions"]])
