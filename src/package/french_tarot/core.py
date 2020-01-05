import time
from typing import List


def create_timestamp():
    return int(time.time() * 1000000)


def merge_files(input_file_paths: List[str], output_file_path: str):
    with open(output_file_path, 'w') as output_file:
        for input_file_path in input_file_paths:
            with open(input_file_path) as input_file:
                output_file.writelines(input_file.readlines())
