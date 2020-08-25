# Copyright 2020 The Google Research Authors.
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

if [[ $# -ne 1 ]]; then
  GPUID=0,1,2,3,4,5,6,7
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA=convai2/opedata_hard/baseline_model/
#DATA=selfplay_opedata/opedata/full/
#DATA=human_opedata/opedata/full/
DATA_ROOT=$PROJECT_ROOT/data/$DATA
TASK_NAME=air_ope

# output
OUTPUT=$PROJECT_ROOT/outputs/debug/$DATA
#OUTPUT=$PROJECT_ROOT/outputs/debug/random/$DATA

# model
MODEL_TYPE=roberta
MODEL_NAME=roberta-base
CACHEDIR=$PROJECT_ROOT/pretrained_model
#CACHEDIR=$PROJECT_ROOT/pretrained_model/random
FIX_BERT=true
SHARE_BERT=true

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=20
SEED=0

ADAM_EPS=1e-6
WARMUP=2000

TRAIN_BATCH=256

MAXLEN=384

# Dice parameter
alphaR=1
alphaC=1
alphaQ=0
regfunC=square
regfunQ=square


[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'data' --exclude 'pretrained_model' --exclude 'outputs' --exclude 'transformers' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
  $($FIX_BERT && echo '--fix_bert') \
  $($SHARE_BERT && echo '--share_bert') \
  --alphaR $alphaR \
  --alphaC $alphaC \
  --alphaQ $alphaQ \
  --regfunC $regfunC \
  --regfunQ $regfunQ \
  --data_dir $DATA_ROOT \
  --task_name $TASK_NAME \
  --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_device_train_batch_size $TRAIN_BATCH \
  --logging_steps 10 \
  --evaluate_during_training \
  --save_steps 100000 \
  --do_train \
  --output_dir $OUTPUT \
  --logging_dir $OUTPUT\log \
  --cache_dir $CACHEDIR \
  --seed $SEED \
  --max_seq_length $MAXLEN \
  --workers 50 \
  --save_embedding \
  --overwrite_output_dir
