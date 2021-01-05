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
# Example of running model training and decoding locally.
# bash run_locally.sh train /tmp/testmodel
# bash run_locally.sh decode /tmp/testmodel

# Assumes presence of ROCSTORIES_DATA and HYEPRS env variables.
if [[ -z $ROCSTORIES_DATA ]]
then
      echo "\$ROCSTORIES_DATA is empty"
      exit
fi
# TODO(peterjliu): Assume HYPERS is in data directory.
if [[ -z $HYPERS ]]
then
      echo "\$HYPERS is empty, should be json file specifying model settings."
      exit
fi

echo "Using data from $ROCSTORIES_DATA and model hypers: $HYPERS"

# ROCSTORIES_DATA should contain:
# Train data: rocstories_springwintertrain.all.00xx.tfrecord where xx=00,..,17
# And the following 3 files.
GT_VALID=${ROCSTORIES_DATA}/rocstories_gt.test.tfrecord
GT_TEST=${ROCSTORIES_DATA}/rocstories_gt.valid.tfrecord
VOCAB_FILE=${ROCSTORIES_DATA}/wikitext103_32768.subword_vocab

BIN="python -m summae.pors"

# Train
function train() {
  MODEL_DIR=$1
  echo "Training for 10 steps from scratch, saving model to $MODEL_DIR."
  $BIN --data_dir=${ROCSTORIES_DATA} --mode=train --logtostderr \
    --params_file=${HYPERS} \
    --train_steps=10  --batch_size=4 --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --task=rocstories
}

function decode() {
  MODEL_DIR=$1
  CHECKPOINT=$2
  echo "Decode model $MODEL_DIR with checkpoint $CHECKPOINT to $MODEL_DIR/decodes"
  # Decode valid GT
  $BIN --data_dir=${ROCSTORIES_DATA} --logtostderr \
    --params_file=${HYPERS} --batch_size=100 --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --task=rocstories --mode=decode \
    --decode_reconstructions --eval_subset=valid_gt,test_gt --eval_checkpoints=$CHECKPOINT
}


CMD=$1
if [ "$CMD" == "train" ]; then
  if [ "$#" -ne 2 ]; then
    echo "Must specify train model_dir"
  fi
  train $2
elif [ "$CMD" == "decode" ]; then
  if [ "$#" -ne 3 ]; then
    echo "Must specify train model_dir"
  fi
  decode $2 $3
else
  echo "Must specify train|decode command"
fi
