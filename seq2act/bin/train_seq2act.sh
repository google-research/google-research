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
DEFINE_string train "ground" "Specify the training type: [parse, ground]"
DEFINE_string hparam_file "./seq2act/ckpt_hparams/grounding/" \
                          "Specify the hyper-parameter file"
DEFINE_string parser_checkpoint "./seq2act/ckpt_hparams/tuple_extract" \
                                "Specify the checkpoint of tuple extraction"
DEFINE_string rico_sca_train "./seq2act/data/rico_sca/tfexample/*[!0].tfrecord" \
                             "Specify the path to rico_sca dataset"
DEFINE_string android_howto_train "./seq2act/data/android_howto/tfexample/*[!0].tfrecord" \
                                  "Specify the path to android_howto dataset"

gbash::init_google "$@"

set -e
set -x

virtualenv -p python3.7 .
source ./bin/activate

pip install tensorflow
pip install -r seq2act/requirements.txt

if (( "${FLAGS_train}" == "parse" )); then
  train_file_list="${FLAGS_android_howto_train},${FLAGS_rico_sca_train}"
  train_batch_sizes="128,128"
  train_source_list="android_howto,rico_sca"
  train_steps=1000000
  python -m seq2act.bin.seq2act_train_eval --exp_mode "train" \
                                           --experiment_dir "${FLAGS_experiment_dir}" \
                                           --hparam_file "${FLAGS_hparam_file}" \
                                           --train_steps "${train_steps}" \
                                           --train_file_list "${train_file_list}" \
                                           --train_batch_sizes "${train_batch_sizes}" \
                                           --train_source_list "${train_source_list}" \
else
  train_file_list="${FLAGS_rico_sca_train}"
  train_batch_sizes="64"
  train_source_list="rico_sca"
  train_steps=250000
  python -m seq2act.bin.seq2act_train_eval --exp_mode "train" \
                                           --experiment_dir "${FLAGS_experiment_dir}" \
                                           --hparam_file "${FLAGS_hparam_file}" \
                                           --train_steps "${train_steps}" \
                                           --train_file_list "${train_file_list}" \
                                           --train_batch_sizes "${train_batch_sizes}" \
                                           --train_source_list "${train_source_list}" \
                                           --reference_checkpoint "${FLAGS_parser_checkpoint}"
fi



