import os

import dill


def load_test_data(file_name):
    path = os.path.join(os.path.dirname(__file__), "resources", file_name)
    with open(path, "rb") as f:
        obj = dill.load(f)
    return obj


def run_test_on_data(stuff, mocker):
    obj = load_test_data(stuff)
    if not hasattr(obj["environment"], '_original_player_ids'):
        obj['environment']._original_player_ids = []
    if not hasattr(obj["environment"], 'n_players'):
        obj['environment'].n_players = 4
    observation, _, done, _ = obj["environment"].step(obj["agent"].get_action(obj["observation"]))


def test_stress_test_iteration_433_wrong_assignment_type(mocker):
    run_test_on_data("stress_test_433.dill", mocker)
