import os

import dill


def load_test_data(file_name):
    path = os.path.join(os.path.dirname(__file__), "resources", file_name)
    with open(path, "rb") as f:
        obj = dill.load(f)
    return obj


def run_test_on_data(stuff):
    obj = load_test_data(stuff)
    observation, _, done, _ = obj["environment"].step(obj["agent"].get_action(obj["observation"]))


def test_stress_test_iteration_21_wrong_assignment_type():
    run_test_on_data("stress_test_433.dill")
