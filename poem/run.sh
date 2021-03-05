# Copyright 2021 The Google Research Authors.
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

#!/bin/bash
#
# Note: This script serves for testing purposes. It also provides a sample
# command for how to launch a training job on a small dummy data table. Running
# it as is does NOT launch training job on any real datasets or generate any
# meaningful model checkpoints.

set -e
set -x

virtualenv -p python3 env
source env/bin/activate
pip3 install -r poem/requirements.txt

TRAIN_DIR="/tmp/e3d/train"

mkdir -p "${TRAIN_DIR}"
python3 -m poem.train \
  --alsologtostderr \
  --input_table="poem/testdata/tfe-2.tfrecords" \
  --train_log_dir="${TRAIN_DIR}" \
  --batch_size=4 \
  --num_steps=5 \
  --input_shuffle_buffer_size=10 \
  --summarize_percentiles=false

assert_file_exists() {
  if [[ ! -f "$1" ]]; then
    echo "File does not exist: $1"
    exit 1
  fi
}

assert_file_exists "${TRAIN_DIR}/all_flags.train.json"
assert_file_exists "${TRAIN_DIR}/graph.pbtxt"
assert_file_exists "${TRAIN_DIR}/model.ckpt-00000005.data-00000-of-00001"
assert_file_exists "${TRAIN_DIR}/model.ckpt-00000005.meta"
assert_file_exists "${TRAIN_DIR}/model.ckpt-00000005.index"

echo "PASS"
