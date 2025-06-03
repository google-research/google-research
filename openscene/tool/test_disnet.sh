#!/bin/sh
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


set -x
set -e

exp_name=$1
config=$2
pred_type=$3
T=$4

export OPENBLAS_NUM_THREADS=${T}
export GOTO_NUM_THREADS=${T}
export OMP_NUM_THREADS=${T}
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
TRAIN_CODE=train.py
TEST_CODE=test_disnet.py


exp_dir=Exp/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

now=$(date +"%Y%m%d_%H%M%S")

cp tool/test_disnet.sh tool/${TEST_CODE} ${exp_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

export PYTHONPATH=.
now=$(date +"%Y%m%d_%H%M%S")
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  pred_type ${pred_type} \
  save_folder ${result_dir}/val \
  model_path ${model_dir}/model_best.pth.tar