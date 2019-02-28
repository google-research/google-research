# Copyright 2019 The Google Research Authors.
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

BEAM_SIZE=5
CONFIG="dev"
EVAL_BATCH_SIZE=50
EVAL_TYPE=eval_only
UNITTEST=nounittest
GPU_ID=0

while [[ "$#" -ge 1 ]]; do case $1 in
    --config) CONFIG="$2";;
    --name) NAME="$2";;
    --gpu) GPU_ID=$2;;
    --beam_size) BEAM_SIZE=$2;;
    --bs) EVAL_BATCH_SIZE=$2;;
    --et) EVAL_TYPE="$2";;
    --test) UNITTEST="$2";;
    *) echo "Usage: $0 --name EXPERIMENT_NAME --config (dev|test) --bs BATCH_SIZE --gpu gpu_id --beam_size BEAM_SIZE"
       exit 1;;
  esac; shift; shift
done

DATA_DIR=$HOME"/projects/data/wikitable/"
INPUT_DIR=$DATA_DIR"processed_input/wtq_preprocess/"
SPLIT_DIR=$INPUT_DIR"data_split_1/"

case $CONFIG in
    dev)
        echo "Evaluate on dev set."
        EVAL_FILE=$SPLIT_DIR"dev_split.jsonl"
        ;;
    test)
        echo "Evaluate on test set!"
        EVAL_FILE=$INPUT_DIR"test_split.jsonl"
        ;;
    *)
	exit 1
        ;;
esac

NAME_EXP=$NAME
if ! [[ $NAME =~ ^eval ]]; then
  NAME_EXP="eval_"$NAME
fi

/usr/bin/stdbuf -oL python -m meta_reward_learning.semantic_parsing.experiment_single \
       --logtostderr \
       --show_log \
       --ckpt_from_another \
       --$EVAL_TYPE \
       --$UNITTEST \
       --load_saved_programs \
       --eval_batch_size=$EVAL_BATCH_SIZE \
       --eval_beam_size=$BEAM_SIZE \
       --output_dir=$DATA_DIR"output" \
       --experiment_to_eval=$NAME \
       --experiment_name=$NAME_EXP"_beam"$BEAM_SIZE \
       --eval_file=$EVAL_FILE \
       --dev_file=$EVAL_FILE \
       --eval_use_gpu \
       --eval_gpu_id=$GPU_ID
