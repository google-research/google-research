# Copyright 2022 The Google Research Authors.
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

assert_file_exists() {
  if [[ ! -f "$1" ]]; then
    echo "File does not exist: $1"
    exit 1
  fi
}

# Check if Pr-VIPE training runs.
PR_VIPE_TRAIN_DIR="/tmp/e3d/pr_vipe/train"

mkdir -p "${PR_VIPE_TRAIN_DIR}"
python3 -m poem.pr_vipe.train \
  --alsologtostderr \
  --input_table="poem/testdata/tfe-2.tfrecords" \
  --train_log_dir="${PR_VIPE_TRAIN_DIR}" \
  --batch_size=4 \
  --num_steps=5 \
  --input_shuffle_buffer_size=10 \
  --summarize_percentiles=false \
  || exit 1

assert_file_exists "${PR_VIPE_TRAIN_DIR}/all_flags.train.json"
assert_file_exists "${PR_VIPE_TRAIN_DIR}/graph.pbtxt"
assert_file_exists "${PR_VIPE_TRAIN_DIR}/model.ckpt-00000005.data-00000-of-00001"
assert_file_exists "${PR_VIPE_TRAIN_DIR}/model.ckpt-00000005.meta"
assert_file_exists "${PR_VIPE_TRAIN_DIR}/model.ckpt-00000005.index"

# Check if Temporal Pr-VIPE training runs.
TEMPORAL_PR_VIPE_TRAIN_DIR="/tmp/e3d/temporal_pr_vipe/train"

mkdir -p "${TEMPORAL_PR_VIPE_TRAIN_DIR}"
python3 -m poem.pr_vipe.temporal.train \
  --alsologtostderr \
  --input_table="poem/testdata/tfse-2.tfrecords" \
  --train_log_dir="${TEMPORAL_PR_VIPE_TRAIN_DIR}" \
  --batch_size=4 \
  --num_steps=5 \
  --input_shuffle_buffer_size=10 \
  --summarize_percentiles=false \
  --input_sequence_length=5 \
  || exit 1

assert_file_exists "${TEMPORAL_PR_VIPE_TRAIN_DIR}/all_flags.train.json"
assert_file_exists "${TEMPORAL_PR_VIPE_TRAIN_DIR}/graph.pbtxt"
assert_file_exists "${TEMPORAL_PR_VIPE_TRAIN_DIR}/model.ckpt-00000005.data-00000-of-00001"
assert_file_exists "${TEMPORAL_PR_VIPE_TRAIN_DIR}/model.ckpt-00000005.meta"
assert_file_exists "${TEMPORAL_PR_VIPE_TRAIN_DIR}/model.ckpt-00000005.index"

# Check if CV-MIM encoder training runs.
CV_MIM_ENCODER_TRAIN_DIR="/tmp/e3d/cv_mim/encoder_train"

mkdir -p "${CV_MIM_ENCODER_TRAIN_DIR}"
python3 -m poem.cv_mim.train \
  --alsologtostderr \
  --log_dir_path="${CV_MIM_ENCODER_TRAIN_DIR}" \
  --input_table="poem/testdata/tfe-2.tfrecords" \
  --shuffle_buffer_size=10 \
  --batch_size=4 \
  --num_iterations=5 \
  || exit 1

assert_file_exists "${CV_MIM_ENCODER_TRAIN_DIR}/all_flags.train.json"
assert_file_exists "${CV_MIM_ENCODER_TRAIN_DIR}/ckpt-5.data-00000-of-00001"
assert_file_exists "${CV_MIM_ENCODER_TRAIN_DIR}/ckpt-5.index"

# Check if CV-MIM downstream training with frozen encoder runs.
CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR="/tmp/e3d/cv_mim/train_with_encoder"

mkdir -p "${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}"
python3 -m poem.cv_mim.action_recognition.train_with_encoder \
  --alsologtostderr \
  --log_dir_path="${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}" \
  --encoder_checkpoint_path="${CV_MIM_ENCODER_TRAIN_DIR}/ckpt-5" \
  --input_tables="poem/testdata/tfe-1-seq.tfrecords" \
  --batch_sizes=4 \
  --shuffle_buffer_size=10 \
  --num_classes=6 \
  --num_frames=5 \
  --num_iterations=5 \
  || exit 1

assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}/all_flags.train_with_encoder.json"
assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}/ckpt-5.data-00000-of-00001"
assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}/ckpt-5.index"

# Check if CV-MIM downstream evaluation with frozen encoder runs.
python3 -m poem.cv_mim.action_recognition.eval_with_encoder \
  --alsologtostderr \
  --eval_name="test_run" \
  --log_dir_path="${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}" \
  --encoder_checkpoint_path="${CV_MIM_ENCODER_TRAIN_DIR}/ckpt-5" \
  --input_tables="poem/testdata/tfe-1-seq.tfrecords" \
  --batch_sizes=4 \
  --num_classes=6 \
  --num_frames=5 \
  --continuous_eval=false \
  || exit 1

assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_ENCODER_DIR}/test_run/all_flags.eval_with_encoder.json"

# Check if CV-MIM downstream training with pre-computed features runs.
CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR="/tmp/e3d/cv_mim/train_with_features"

mkdir -p "${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}"
python3 -m poem.cv_mim.action_recognition.train_with_features \
  --alsologtostderr \
  --log_dir_path="${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}" \
  --input_tables="poem/testdata/tfe-1-seq.tfrecords" \
  --batch_sizes=4 \
  --input_features_dim=32 \
  --shuffle_buffer_size=10 \
  --num_classes=6 \
  --num_frames=5 \
  --num_iterations=5 \
  || exit 1

assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}/all_flags.train_with_features.json"
assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}/ckpt-5.data-00000-of-00001"
assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}/ckpt-5.index"

# Check if CV-MIM downstream evaluation with pre-computed features runs.
python3 -m poem.cv_mim.action_recognition.eval_with_features \
  --alsologtostderr \
  --eval_name="test_run" \
  --log_dir_path="${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}" \
  --input_tables="poem/testdata/tfe-1-seq.tfrecords" \
  --batch_sizes=4 \
  --input_features_dim=32 \
  --num_classes=6 \
  --num_frames=5 \
  --continuous_eval=false \
  || exit 1

assert_file_exists "${CV_MIM_DOWNSTREAM_TRAIN_WITH_FEATURES_DIR}/test_run/all_flags.eval_with_features.json"

echo "PASS"
