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
source gbash.sh || exit

DEFINE_string experiment_dir --required "" "Specify the experimental directory"
DEFINE_string eval_files "./seq2act/data/rico_sca/*0.tfrecord" \
                          "Specify the path to the eval dataset"
DEFINE_string eval_data_source "rico_sca" "Specify eval data source"
DEFINE_string eval_name "rico_sca" "Specify eval job name"
DEFINE_string metric_types "final_accuracy,ref_accuracy,basic_accuracy" \
                           "Specify the eval metric types"
DEFINE_int eval_steps 200 "Specify the eval steps"
DEFINE_int eval_batch_size 16 "Specify the eval batch size"
DEFINE_int decode_length 20 "Specify the decode length"

gbash::init_google "$@"

set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install -r seq2act/requirements.txt

python -m seq2act.bin.seq2act_train_eval --exp_mode "eval" \
                                         --experiment_dir "${FLAGS_experiment_dir}" \
                                         --eval_files "${FLAGS_eval_files}" \
                                         --metric_types "${FLAGS_metric_types}" \
                                         --decode_length "${FLAGS_decode_length}" \
                                         --eval_name "${FLAGS_eval_name}" \
                                         --eval_steps "${FLAGS_eval_steps}" \
                                         --eval_data_source "${FLAGS_eval_data_source}" \
                                         --eval_batch_size ${FLAGS_eval_batch_size}
