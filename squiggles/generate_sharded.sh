#!/bin/bash
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Generate a Squiggles dataset that is easy to read using tfds.
#
# First, run generate_shard many times in parallel to generate all the test and
# train shards. Then, run generate_metadata to generate the metadata that will
# make the dataset very easy to load using tensorflow_datasets, as demonstrated
# in load_data_example.py.
#
# Options needed:
#
# * base_path
# * num_train_shards
# * num_test_shards
# * samples_per_shard
# * squiggle_algorithm
# * (optional) generate_shard command(s). Defaults to
#   "python3 generate_shard.py".
# * (optional) generate_metadata command(s). Defaults to
#   "python3 generate_metadata.py".

PYTHON=$(which python3 || which python || which py)

BASE_PATH="$1"
NUM_TRAIN_SHARDS="$2"
NUM_TEST_SHARDS="$3"
SAMPLES_PER_SHARD="$4"
SQUIGGLE_ALGORITHM="$5"
EXEC_GENERATE_SHARD="${6:-${PYTHON} -m squiggles.generate_shard}"
EXEC_GENERATE_META="${7:-${PYTHON} -m squiggles.generate_metadata}"

FILENAME=$(basename "$BASE_PATH")
if [[ "$FILENAME" =~ [A-Z_-] ]];
then
  echo "$FILENAME cannot be used as the basis for file names. Underscores, hyphens and captilized letters are not supported by tfds in the base file name."
  exit 1
fi

# Ensure Ctrl-C terminates all child processes
# from https://stackoverflow.com/a/35660327/2318074
trap terminate SIGINT
terminate(){
    pkill -SIGINT -P $$
    exit
}

# Generate training shards
for i in $(seq 0 $(( NUM_TRAIN_SHARDS - 1 ))); do
  ${EXEC_GENERATE_SHARD} \
    --base_path="${BASE_PATH}" \
    --split_name=train \
    --shard_num="${i}" \
    --num_shards="${NUM_TRAIN_SHARDS}" \
    --samples_per_shard="${SAMPLES_PER_SHARD}" \
    --squiggle_algorithm="${SQUIGGLE_ALGORITHM}" &
done

# Generate testing shards
for i in $(seq 0 $(( NUM_TEST_SHARDS - 1 ))); do
  ${EXEC_GENERATE_SHARD} \
    --base_path="${BASE_PATH}" \
    --split_name=test \
    --shard_num="${i}" \
    --num_shards="${NUM_TEST_SHARDS}" \
    --samples_per_shard="${SAMPLES_PER_SHARD}" \
    --squiggle_algorithm="${SQUIGGLE_ALGORITHM}" &
done

# Wait until all the shards are finished.
wait
echo "All shards written; now generating metadata."

# Generate the metadata file
${EXEC_GENERATE_META} \
  --base_path="${BASE_PATH}" \
  --squiggle_algorithm="${SQUIGGLE_ALGORITHM}" \
  --samples_per_shard="${SAMPLES_PER_SHARD}" \
  --num_train_shards="${NUM_TRAIN_SHARDS}" \
  --num_test_shards="${NUM_TEST_SHARDS}"

echo "generate_sharded done"
