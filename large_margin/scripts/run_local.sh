#!/bin/bash

# train or eval.
JOB=${1}

EXP="mnist"

# Specify data directory below.
DATA_DIR=""

if [[ "${JOB}" == "train" ]]
then
  python ../train.py \
  --experiment_type="${EXP}" \
  --batch_size=128 \
  --num_epochs=300 \
  --data_dir="${DATA_DIR}" \
  --logtostderr

elif [[ "${JOB}" == "eval" ]]
then
  python ../eval.py \
  --experiment_type="${EXP}" \
  --batch_size=10 \
  --data_dir="${DATA_DIR}" \
  --logtostderr
fi
