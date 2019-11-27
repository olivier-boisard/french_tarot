import json
import os
import subprocess
from dataclasses import dataclass
from typing import List, Dict

from french_tarot.environment import french_tarot
from french_tarot.environment.core import CARDS
from french_tarot.reagent.data import ReAgentDataRow


def convert_to_timeline_format(batch: List[ReAgentDataRow], output_folder: str):
    docker_workdir = _get_docker_working_directory()
    reagent_dir = _get_reagent_directory()
    input_table_name = "french_tarot_discrete"
    table_path = os.path.join(reagent_dir, input_table_name)

    _dump_batch_to_json_file(batch, table_path)

    docker_command = [
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--volume={}:{}".format(docker_workdir, docker_workdir),
        "--workdir={}".format(os.path.join(docker_workdir, "ReAgent")),
        "--publish=0.0.0.0:6006:6006",
        "french_tarot:latest"
    ]
    spark_command = [
        os.path.join("/", "usr", "local", "spark", "bin", "spark-submit"),
        "--class=com.facebook.spark.rl.Preprocessor",
        "preprocessing/target/rl-preprocessing-1.1.jar",
        "{}".format(json.dumps(vars(_generate_timeline(batch, input_table_name))))
    ]

    subprocess.call(docker_command + spark_command)


def _get_reagent_directory():
    return os.path.join(_get_docker_working_directory(), "ReAgent")


def _get_docker_working_directory():
    docker_workdir = os.path.join(os.path.dirname(french_tarot.__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    return os.path.abspath(docker_workdir)


def _generate_timeline(batch: List[ReAgentDataRow], input_table_name: str) -> 'Timeline':
    dataset_ids = [row.ds for row in batch]
    output_dir = os.path.dirname(input_table_name)

    # We use a dict instead of a dataclass here because attribute names must be lowerCamelCase and this would break
    # PEP8
    timeline = {
        "startDs": min(dataset_ids),
        "endDs": max(dataset_ids),
        "addTerminalStateRow": True,  # final row in each MDP is corresponds to the terminal state
        "actionDiscrete": True,
        "inputTableName": input_table_name,
        "outputTableName": input_table_name + "_training",
        "evalTableName": input_table_name + "_eval",
        "numOutputShards": 1,  # some Spark stuff that should be kept to 1 in case of one machine only process
    }
    query = {
        "tableSample": 100,
        "actions": [str(i) for i in range(len(CARDS))]
    }
    return Timeline(timeline=timeline, query=query)


def _dump_batch_to_json_file(batch, output_filename):
    json_objects = map(lambda row: json.dumps(row.dictionary), batch)
    print("Dump json object at", output_filename)
    with open(output_filename, "w") as f:
        f.writelines(json_objects)


@dataclass
class Timeline:
    timeline: Dict
    query: Dict
