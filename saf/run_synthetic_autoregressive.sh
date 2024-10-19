# Copyright 2024 The Google Research Authors.
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

set -e

mkdir -p logs

EXPERIMENT_VERSION=v0

SYN_OPTION=1
DATASET=synthetic_autoregressive
for LEN_TOTAL in {1000..4000..1000}
do
  MODEL_NAME=lstm_seq2seq
  FILENAME=logs/experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}_${SYN_OPTION}_${LEN_TOTAL}-out.log
  touch ${FILENAME}
  FILESIZE=$(stat -c%s ${FILENAME})
  if ((${FILESIZE} < 2000)) ; then
      nohup python3 -m experiment_${DATASET} --model_type=${MODEL_NAME} --gpu_index=0 --synthetic_data_option=${SYN_OPTION} --len_total=${LEN_TOTAL} --filename=experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}>${FILENAME}&
  fi
  MODEL_NAME=lstm_seq2seq_saf
  FILENAME=logs/experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}_${SYN_OPTION}_${LEN_TOTAL}-out.log
  touch ${FILENAME}
  FILESIZE=$(stat -c%s ${FILENAME})
  if ((${FILESIZE} < 2000)) ; then
      nohup python3 -m experiment_${DATASET} --model_type=${MODEL_NAME} --gpu_index=1 --synthetic_data_option=${SYN_OPTION} --len_total=${LEN_TOTAL} --filename=experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}>${FILENAME}&
  fi
  MODEL_NAME=tft
  FILENAME=logs/experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}_${SYN_OPTION}_${LEN_TOTAL}-out.log
  touch ${FILENAME}
  FILESIZE=$(stat -c%s ${FILENAME})
  if ((${FILESIZE} < 2000)) ; then
      nohup python3 -m experiment_${DATASET} --model_type=${MODEL_NAME} --gpu_index=2 --synthetic_data_option=${SYN_OPTION} --len_total=${LEN_TOTAL} --filename=experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}>${FILENAME}&
  fi
  MODEL_NAME=tft_saf
  FILENAME=logs/experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}_${SYN_OPTION}_${LEN_TOTAL}-out.log
  touch ${FILENAME}
  FILESIZE=$(stat -c%s ${FILENAME})
  if ((${FILESIZE} < 2000)) ; then
      nohup python3 -m experiment_${DATASET} --model_type=${MODEL_NAME} --gpu_index=3 --synthetic_data_option=${SYN_OPTION} --len_total=${LEN_TOTAL} --filename=experiment_${DATASET}_${MODEL_NAME}_${EXPERIMENT_VERSION}>${FILENAME}&
  fi
  while [ `jobs | wc -l` -ge 6 ] ; do
      sleep 1
  done
done
