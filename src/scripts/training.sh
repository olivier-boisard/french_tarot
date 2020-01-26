#!/bin/bash

repo_dir=$(git rev-parse --show-toplevel)
python_exec=$repo_dir/venv/bin/python
python_workdir=$repo_dir/src/package
training_folder_tmp=$repo_dir/tmp
generation_module=french_tarot.applications.generate_card_phase_data
timeline_filepath="$training_folder_tmp/french_tarot"

echo "Delete previous session tmp folder"
rm -rf $training_folder_tmp

echo "Generate data"
mkdir -p "$training_folder_tmp"
(cd "$python_workdir" && $python_exec -m $generation_module $timeline_filepath --n-max-episodes=10)

echo "Prepare data for training"
spark_command="/usr/local/spark/bin/spark-submit --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \"\`cat $repo_dir/timeline.json\`\""
merge_training_command="cat french_tarot_training/part* > $training_folder_tmp/french_tarot_training.json"
merge_eval_command="cat french_tarot_eval/part* > $training_folder_tmp/french_tarot_eval.json"
docker run \
  --runtime=nvidia \
  --volume=$repo_dir:$repo_dir \
  --workdir=/opt/ReAgent \
  --rm \
  french_tarot:latest \
  /bin/bash -c "cp $timeline_filepath /opt/ReAgent && $spark_command && $merge_training_command && $merge_eval_command"
