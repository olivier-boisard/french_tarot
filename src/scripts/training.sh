#!/bin/bash

repo_dir=$(git rev-parse --show-toplevel)
python_exec=$repo_dir/venv/bin/python
python_workdir=$repo_dir/src/package
training_folder_tmp=$repo_dir/tmp

generation_module=french_tarot.applications.generate_card_phase_data
mkdir -p "$training_folder_tmp"
(cd "$python_workdir" && $python_exec -m $generation_module "$training_folder_tmp/samples.json" --n-max-episodes=10000)
rm -rf "$training_folder_tmp"
