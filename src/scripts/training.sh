#!/bin/bash

repo_dir=$(git rev-parse --show-toplevel)
python_exec=$repo_dir/venv/bin/python
python_workdir=$repo_dir/src/package
training_folder_tmp=$repo_dir/tmp
reagent_folder=$repo_dir/ReAgent
generation_module=french_tarot.applications.generate_card_phase_data
timeline_filepath="$training_folder_tmp/samples.json"

echo "Delete previous session tmp folder"
rm -rf $training_folder_tmp

echo "Generate data"
mkdir -p "$training_folder_tmp"
(cd "$python_workdir" && $python_exec -m $generation_module "$timeline_filepath" --n-max-episodes=10)

echo "Prepare data for training"
(
  cd "$reagent_folder" && docker run \
    --runtime=nvidia \
    --volume="$reagent_folder":"$reagent_folder" \
    --volume=$training_folder_tmp:$training_folder_tmp \
    --workdir="$reagent_folder" \
    --rm \
    french_tarot:latest \
    /usr/local/spark/bin/spark-submit \
    --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \
    $(cat $timeline_filepath)
)
