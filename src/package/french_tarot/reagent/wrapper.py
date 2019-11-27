import json
import os
from dataclasses import dataclass
from typing import List, Dict

from french_tarot.environment import french_tarot
from french_tarot.environment.core import CARDS
from french_tarot.reagent.data import ReAgentDataRow


def convert_to_timeline_format(batch: List[ReAgentDataRow], output_folder: str):
    docker_workdir = os.path.join(os.path.dirname(french_tarot.__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    docker_workdir = os.path.abspath(docker_workdir)

    docker_command = [
        "docker run",
        "--rm",
        "--runtime-nvidia",
        "-v {}:{}".format(docker_workdir, docker_workdir),
        "-w {}".format(os.path.join(docker_workdir, "ReAgent")),
        "-p 0.0.0.0:6006:6006",
        "horizon:dev"
    ]

    docker_run_command = [
        os.path.join("/", "usr", "local", "spark", "bin", "spark-submit"),
        "--class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar",
        "{}".format(json.dumps(generate_timeline(batch)))
    ]


def generate_timeline(batch: List[ReAgentDataRow]) -> 'Timeline':
    dataset_ids = [row.ds for row in batch]

    # We use a dict instead of a dataclass here because attribute names must be lowerCamelCase and this would break
    # PEP8
    timeline = {
        "startDs": min(dataset_ids),
        "endDs": max(dataset_ids),
        "addTerminalStateRow": True,  # final row in each MDP is corresponds to the terminal state
        "actionDiscrete": True,
        "inputTableName": "french_tarot_discrete",
        "outputTableName": "french_tarot_discrete_training",
        "evalTableName": "french_tarot_discrete_eval",
        "numOutputShards": 1,  # some Spark stuff that should be kept to 1 in case of one machine only process
    }
    query = {
        "tableSample": 100,
        "actions": [str(i) for i in range(len(CARDS))]
    }
    return Timeline(timeline=timeline, query=query)


@dataclass
class Timeline:
    timeline: Dict
    query: Dict
