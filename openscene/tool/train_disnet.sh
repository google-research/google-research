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
T=$3

export OPENBLAS_NUM_THREADS=${T}
export GOTO_NUM_THREADS=${T}
export OMP_NUM_THREADS=${T}
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
TRAIN_CODE=train_disnet.py

exp_dir=Exp/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train_disnet.sh tool/${TRAIN_CODE} ${config} ${exp_dir}

export PYTHONPATH=.
#rm -rf /dev/shm/wbhu*
echo $OMP_NUM_THREADS | tee -a ${exp_dir}/train-$now.log
nvidia-smi | tee -a ${exp_dir}/train-$now.log
which pip | tee -a ${exp_dir}/train-$now.log

$PYTHON -u ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$now.log

# $PYTHON -u ${exp_dir}/${TRAIN_CODE} \
#   --config=${config} \
#   save_path ${exp_dir}

# kernprof -l -v ${exp_dir}/${TRAIN_CODE} \
#   --config=${config} \
#   save_path ${exp_dir}