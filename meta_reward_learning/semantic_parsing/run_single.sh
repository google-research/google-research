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

#!/bin/bash
EXP_TO_LOAD=""
buffer_program_files=""
NACTORS=1
TRAIN_GPU=0
N_EXPLORE=0
N_REPLAY=5
BS=125
SAVE_EVERY_N=100
DRY_RUN=dry_run
OPT='adam'
EVAL_GPU=-1
SHOW_LOG=show_log
UNITTEST=nounittest
DEBUG=nodebug_model
META_LEARN=nometa_learn  # FLAG FOR META_LEARNING
SHARD_END=90
EN_MAXLEN=45
MAX_PROGRAMS=50
SHARD_START=0
ENTROPY_REG=0.0
lr=0.001
meta_lr=0.1
TRAINABLE_ONLY=trainable_only
N_STEPS=20000 # Greater than 17560
INITIAL_BUFFER_FILE="full_mapo_iml_buffer"
VAL_BUFFER_FILE="all_val_programs"
VAL_BS=512
SCORE_TEMP=1.0
SAMPLING_STRATEGY="probs"
PLOT_SUMMARIES="plot_summaries"

while [[ "$#" > 1 ]]; do case $1 in
    --config) CONFIG="$2";;
    --name) NAME="$2";;
    --train_gpu) TRAIN_GPU=$2;;
    --eval_gpu) EVAL_GPU=$2;;
    --extra_buffer) buffer_program_files="$2";;
    --lr) lr=$2;;
    --exp_to_load) EXP_TO_LOAD="$2";;
    --n_actors) NACTORS=$2;;
    --batch_size) BS=$2;;
    --val_batch_size) VAL_BS=$2;;
    --n_replay) N_REPLAY=$2;;
    --log) SHOW_LOG="$2";;
    --shard_end) SHARD_END=$2;;
    --shard_start) SHARD_START=$2;;
    --n_steps) N_STEPS=$2;;
    --ent_reg) ENTROPY_REG=$2;;
    --test) UNITTEST="$2";;
    --debug) DEBUG="$2";;
    --save_every_n) SAVE_EVERY_N=$2;;
    --dry_run) DRY_RUN="$2";;
    --n_explore) N_EXPLORE=$2;;
    --meta_lr) meta_lr=$2;;
    --meta_learn) META_LEARN="$2";;
    --init_buffer) INITIAL_BUFFER_FILE="$2";;
    --val_buffer) VAL_BUFFER_FILE="$2";;
    --max_programs) MAX_PROGRAMS=$2;;
    --sampling) SAMPLING_STRATEGY="$2";;
    --plot_summaries) PLOT_SUMMARIES="$2";;
    *) echo "Incorrect flag $1 Passed"
       exit 1;;
  esac; shift; shift
done

DATA_DIR=$HOME"/projects/data/wikitable/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_14/"
SPLIT_DIR=$INPUT_DIR"data_split_1/"
INITIAL_BUFFER_FILE=$DATA_DIR"processed_input/"$INITIAL_BUFFER_FILE".json"
VAL_BUFFER_FILE=$DATA_DIR"processed_input/"$VAL_BUFFER_FILE".json"
p_files=""
if [ -n "$buffer_program_files" ]; then
  IFS=',' read -ra ADDR <<< "$buffer_program_files"
  for i in "${ADDR[@]}"; do
    p_files=${p_files}$DATA_DIR"processed_input/"${i}"_buffer_programs.json,"
  done
  p_files=${p_files::-1}
fi

LOGFILE=$HOME"/projects/logs/single_runs/wikitable/"$NAME".log"
if [ $EVAL_GPU -eq -1 ]; then
  EVAL_GPU=$(( TRAIN_GPU + 1 ))
fi
EVAL_GPU=$(( EVAL_GPU%4 ))
echo ""
echo "CONFIG: $CONFIG, name: $NAME, TRAIN_GPU: $TRAIN_GPU, ACTORS: $NACTORS"
echo "ENT_REG: $ENTROPY_REG, SHARD_END:$SHARD_END"
echo "NSTEPS: $N_STEPS, BS: $BS SHARD_START: $SHARD_START N_EXPLORE: $N_EXPLORE N_REPLAY:$N_REPLAY"
echo "EXTRA BUFFER: $p_files EVAL_GPU: $EVAL_GPU LR: $lr META_LR: $meta_lr"
echo "EXP_TO_LOAD: $EXP_TO_LOAD CKPT_EVERY:$SAVE_EVERY_N MAX_PROGS: $MAX_PROGRAMS"
echo "FLAGS: --$DEBUG --$UNITTEST --$SHOW_LOG --$META_LEARN"
echo "LOGFILE: $LOGFILE"
echo "BUFFER: $INITIAL_BUFFER_FILE"
echo "VAL_BUFFER: $VAL_BUFFER_FILE, VAL_BS: $VAL_BS"
echo "--$PLOT_SUMMARIES, SAMPLING: $SAMPLING_STRATEGY"

if [[ "$DRY_RUN" == "dry_run" ]]; then
  echo "Exiting now!"
  exit 0
fi

USE_NONREPLAY=nouse_nonreplay_samples_in_train
RANDOM_REPLAY=norandom_replay_samples
USE_REPLAY_PROB=nouse_replay_prob_as_weight
FIXED_REPLAY_WEIGHT=1.0
TOPK_REPLAY=nouse_top_k_replay_samples
USE_TRAINER_PROB=nouse_trainer_prob
TRUNCATE_AT_N=0
OUTPUT=output
case $CONFIG in
    mapo)
        echo mapo
        #USE_NONREPLAY=use_nonreplay_samples_in_train
        USE_REPLAY_PROB=use_replay_prob_as_weight
        ENTROPY_REG=0.01
        #IN_REPLAY=1
        ;;
    mml)
        echo mml
        ;;
    iml)
        echo iml
        RANDOM_REPLAY=random_replay_samples
        ;;
    hard_em)
        echo hard_em
        TOPK_REPLAY=use_top_k_replay_samples
        N_REPLAY=1
        BS=100
        ;;
    *)
        echo "Usage: $0 (mapo|mapo_enum|mml|iml|hard_em) experiment_name"
        exit 1
        ;;
esac

/usr/bin/stdbuf -oL python -m meta_reward_learning.semantic_parsing.experiment_single  \
       --logtostderr \
       --output_dir=$DATA_DIR$OUTPUT \
       --experiment_name=$NAME \
       --n_actors=$NACTORS \
       --dev_file=$SPLIT_DIR"dev_split.jsonl" \
       --train_shard_dir=$SPLIT_DIR \
       --train_shard_prefix="train_split_shard_90-" \
       --shard_start=$SHARD_START \
       --shard_end=$SHARD_END \
       --load_saved_programs \
       --saved_program_file=$INITIAL_BUFFER_FILE \
       --saved_val_program_file=$VAL_BUFFER_FILE \
       --saved_replay_program_files="$p_files" \
       --embedding_file=$DATA_DIR"raw_input/wikitable_glove_embedding_mat.npy" \
       --vocab_file=$DATA_DIR"raw_input/wikitable_glove_vocab.json" \
       --table_file=$INPUT_DIR"tables.jsonl" \
       --en_vocab_file=$INPUT_DIR"en_vocab_min_count_5.json" \
       --save_every_n=$SAVE_EVERY_N \
       --n_explore_samples=$N_EXPLORE \
       --use_cache \
       --batch_size=$BS \
       --dropout=0.2 \
       --hidden_size=200 \
       --attn_size=200 \
       --attn_vec_size=200 \
       --en_embedding_size=200 \
       --en_bidirectional \
       --n_layers=2 \
       --en_n_layers=2 \
       --use_pretrained_embeddings \
       --pretrained_embedding_size=300 \
       --value_embedding_size=300 \
       --learning_rate=$lr \
       --meta_lr=$meta_lr \
       --n_policy_samples=1 \
       --n_replay_samples=$N_REPLAY \
       --use_replay_samples_in_train \
       --$USE_NONREPLAY \
       --$USE_REPLAY_PROB \
       --$TOPK_REPLAY \
       --fixed_replay_weight=$FIXED_REPLAY_WEIGHT \
       --$RANDOM_REPLAY \
       --min_replay_weight=0.1 \
       --truncate_replay_buffer_at_n=$TRUNCATE_AT_N \
       --train_use_gpu \
       --train_gpu_id=$TRAIN_GPU \
       --eval_use_gpu \
       --eval_gpu_id=$EVAL_GPU \
       --max_n_mem=60 \
       --max_n_valid_indices=60 \
       --entropy_reg_coeff=$ENTROPY_REG \
       --n_steps=$N_STEPS \
       --experiment_to_load=$EXP_TO_LOAD \
       --en_maxlen=$EN_MAXLEN \
       --optimizer=$OPT \
       --max_programs=$MAX_PROGRAMS \
       --$UNITTEST \
       --$DEBUG \
       --show_log \
       --$META_LEARN \
       --score_norm_fn='identity' \
       --score_model='linear' \
       --$TRAINABLE_ONLY \
       --val_batch_size=$VAL_BS \
       --$PLOT_SUMMARIES \
       --sampling_strategy=$SAMPLING_STRATEGY \
       --score_temperature=$SCORE_TEMP \
       --n_extra_explore_for_hard=$N_EXPLORE \
       --ckpt_from_another \
       --use_validation_for_meta_train
