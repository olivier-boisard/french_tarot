import glob
import json
import os
import subprocess
from dataclasses import dataclass
from typing import List, Dict

from french_tarot.core import merge_files
from french_tarot.environment import french_tarot
from french_tarot.environment.core import CARDS
from french_tarot.reagent.data import ReAgentDataRow


# TODO convert to class
def convert_to_timeline_format(batch: List[ReAgentDataRow], output_folder: str, table_sample: int = 100):
    docker_workdir = _get_docker_working_directory()
    reagent_dir = _get_reagent_directory()
    input_table_name = "french_tarot_discrete"
    table_path = os.path.join(reagent_dir, input_table_name)

    _dump_batch_to_json_file(batch, table_path)

    _cleanup_local_spark_cluster(docker_workdir)
    _run_preprocessor(batch, docker_workdir, input_table_name, table_sample)

    os.makedirs(output_folder, exist_ok=True)
    _merge_generated_files(input_table_name, 'training', output_folder)
    _merge_generated_files(input_table_name, 'eval', output_folder)

    # TODO Remove the output data folder
    # rm -Rf cartpole_discrete_training cartpole_discrete_eval


def _merge_generated_files(table_name, step, output_folder):
    input_filepaths = glob.glob(os.path.join(_get_reagent_folder(), table_name + '_' + step, 'part*'))
    output_filepath = os.path.join(_get_reagent_folder(), output_folder, table_name + '_timeline_' + step + '.json')
    merge_files(input_filepaths, output_filepath)


def _get_reagent_folder():
    repository_path = os.path.join(os.path.dirname(french_tarot.__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    repository_realpath = os.path.realpath(repository_path)
    reagent_folder = os.path.join(repository_realpath, 'ReAgent')
    return reagent_folder


def _cleanup_local_spark_cluster(docker_workdir):
    docker_command = _generate_docker_command(docker_workdir)
    cleanup_command = [
        "rm",
        "-rf",
        "spark-warehouse",
        "derby.log",
        "metastore_db",
        "preprocessing/spark-warehouse",
        "preprocessing/metastore_db",
        "preprocessing/derby.log"
    ]
    subprocess.call(docker_command + cleanup_command)


def _run_preprocessor(batch, docker_workdir, input_table_name, table_sample):
    docker_command = _generate_docker_command(docker_workdir)
    spark_command = [
        os.path.join("/", "usr", "local", "spark", "bin", "spark-submit"),
        "--class=com.facebook.spark.rl.Preprocessor",
        "preprocessing/target/rl-preprocessing-1.1.jar",
        "{}".format(json.dumps(vars(_generate_timeline(batch, input_table_name, table_sample))))
    ]
    subprocess.call(docker_command + spark_command)


def _generate_docker_command(docker_workdir):
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
    return docker_command


def _get_reagent_directory():
    return os.path.join(_get_docker_working_directory(), "ReAgent")


def _get_docker_working_directory():
    docker_workdir = os.path.join(os.path.dirname(french_tarot.__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    return os.path.abspath(docker_workdir)


def _generate_timeline(batch: List[ReAgentDataRow], input_table_name: str, table_sample: int) -> 'Timeline':
    dataset_ids = [row.ds for row in batch]

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
        "tableSample": table_sample,
        "actions": [str(i) for i in range(len(CARDS))]
    }
    return Timeline(timeline=timeline, query=query)


def _dump_batch_to_json_file(batch, output_filename):
    json_objects = [json.dumps(row.dictionary) for row in batch]
    print("Dump json object at", output_filename)
    with open(output_filename, "w") as f:
        f.writelines("%s\n" % line for line in json_objects)


@dataclass
class Timeline:
    timeline: Dict
    query: Dict
