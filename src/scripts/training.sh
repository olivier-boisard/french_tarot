#!/bin/bash

repo_dir=$(git rev-parse --show-toplevel)
python_exec=$repo_dir/venv/bin/python
python_workdir=$repo_dir/src/package

generation_module=french_tarot.applications.generate_card_phase_data
(cd "$python_workdir" && $python_exec -m $generation_module samples.json --n-max-episodes=10000)
