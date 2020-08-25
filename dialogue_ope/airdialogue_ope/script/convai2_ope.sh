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

GPUID=0,1,2,3,4,5,6,7
if [ $# -ne 1 ]; then
  GPUID=$GPUID
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA=baseline_model
SUBTASK=""
DATA_ROOT=$PROJECT_ROOT/data/convai2/opedata${SUBTASK}/$DATA/
TASK_NAME=convai2_ope_all

# model
MODEL_TYPE=roberta
MODEL_NAME=roberta-base
FIX_BERT=false
SHARE_BERT=true
FREEZE_BERT=true

# Optimizer params
OPT=adam
LR=1e-4
LR_SCHEDULE='invsqrt'
WEIGHT_DECAY=1e-4
EPOCH=300
SEED=0

ADAM_EPS=1e-6
SGD_MOM=0.5
WARMUP=30

TRAIN_BATCH=20
GRAD_ACCU=1

MAXLEN=400
MAXNORM=1

LOG_STEP=17

# Dice parameter
gamma=1
lambinit=-0.01
alphaR=0
alphaQ=0
alphaC=1
alphaL=0
alphaAux=0
regfunC=square
regfunQ=abs_cut20
regfunL=square
actC=square
actQ=no

# LR scale
LR_C=1
LR_BERT=1
LR_LAMB=10
SCALE_LAMB=100
LR_Q=2

# TAG
TAG=""

# output
OUTPUT=$PROJECT_ROOT/outputs/${TASK_NAME}${SUBTASK}/${DATA}/${MODEL_NAME}_fix_${FIX_BERT}_share_${SHARE_BERT}_freeze_${FREEZE_BERT}_epoch_${EPOCH}_${LR_SCHEDULE}_${OPT}_lr_${LR}_C_${LR_C}_Q_${LR_Q}_L_${LR_LAMB}x${SCALE_LAMB}_BERT_${LR_BERT}_warmup_${WARMUP}_mom_${SGD_MOM}_MAXNORM_${MAXNORM}_WD_${WEIGHT_DECAY}_BS_${TRAIN_BATCH}x${GRAD_ACCU}_Linit_${lambinit}_alphaR_${alphaR}_C_${alphaC}_Q_${alphaQ}_L_${alphaL}_A_${alphaAux}_regfunC_${regfunC}_Q_${regfunQ}_L_${regfunL}_actC_${actC}_Q_${actQ}_tag_${TAG}_seed_${SEED}/
#OUTPUT=$PROJECT_ROOT/outputs/debug

echo "=======OUTDIR======> "
echo $OUTPUT

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'data' --exclude 'pretrained_model' --exclude 'outputs' --exclude 'transformers' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
  $($FIX_BERT && echo '--fix_bert') \
  $($SHARE_BERT && echo '--share_bert') \
  $($FREEZE_BERT && echo '--freeze_bert') \
  --optimizer $OPT \
  --lr_schedule $LR_SCHEDULE \
  --lrscale_c $LR_C \
  --lrscale_q $LR_Q \
  --lrscale_lamb $LR_LAMB \
  --scale_lamb $SCALE_LAMB \
  --lrscale_bert $LR_BERT \
  --lambinit $lambinit \
  --gamma $gamma \
  --alphaR $alphaR \
  --alphaC $alphaC \
  --alphaQ $alphaQ \
  --alphaL $alphaL \
  --alphaAux $alphaAux \
  --regfunC $regfunC \
  --regfunQ $regfunQ \
  --regfunL $regfunL \
  --finalact_c $actC \
  --finalact_q $actQ \
  --data_dir $DATA_ROOT \
  --task_name $TASK_NAME \
  --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --sgd_momentum $SGD_MOM \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_device_train_batch_size $TRAIN_BATCH \
  --logging_steps $LOG_STEP \
  --evaluate_during_training \
  --save_steps 100000 \
  --do_train \
  --output_dir $OUTPUT \
  --logging_dir $OUTPUT\log \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --seed $SEED \
  --max_seq_length $MAXLEN \
  --max_grad_norm $MAXNORM \
  --max_turns 8 \
  --workers 10 \
  --gradient_accumulation_steps $GRAD_ACCU \
  --overwrite_output_dir
