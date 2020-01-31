#!/bin/bash

repo_dir=$(git rev-parse --show-toplevel)
config_dir=$repo_dir/config
python_exec=$repo_dir/venv/bin/python
python_workdir=$repo_dir/src/package
training_folder_tmp=$repo_dir/tmp
generation_module=french_tarot.applications.generate_card_phase_data
timeline_filepath="$training_folder_tmp/french_tarot"
reagent_dir=/opt/ReAgent
docker_run="docker run --runtime=nvidia --volume=$repo_dir:$repo_dir --workdir=$reagent_dir --rm french_tarot:latest"

echo "Delete previous session tmp folder"
rm -rf $training_folder_tmp

echo "Generate data"
mkdir -p "$training_folder_tmp"
(cd "$python_workdir" && $python_exec -m $generation_module $timeline_filepath --n-max-episodes=100000)

echo "Prepare data for training"
preprocessing_command="/usr/local/spark/bin/spark-submit --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \"\`cat $config_dir/timeline.json\`\""
merge_training_command="cat french_tarot_training/part* > $training_folder_tmp/french_tarot_training.json"
merge_eval_command="cat french_tarot_eval/part* > $training_folder_tmp/french_tarot_eval.json"
$docker_run /bin/bash -c "cp $timeline_filepath $reagent_dir && $preprocessing_command && $merge_training_command && $merge_eval_command"

echo "Create normalization parameters"
$docker_run /bin/bash -c "cp $training_folder_tmp/french_tarot_* $reagent_dir && mkdir training_data outputs && python ml/rl/workflow/create_normalization_metadata.py -p $config_dir/dqn.json && mv training_data/ $training_folder_tmp"
