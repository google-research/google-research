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

DATA_DIR="./data/"
DATA_SELECT_DIR="./data/"
MODEL_DIR="./models/"
FT_DIR="./ft/"
SCORES_DIR="./scores/"
BERT_BASE="./BERT/"  # See readme for instructions to download German Bert
BERT_TRAINED="./BERT/trained" # Finetune German BERT on mix of target data
TARGET_TEXT="./target/data.txt" # Read out dataset targets to a text file

# Prepare data
python prepare_data.py --data_dir $DATA_DIR

# Pretrain model
python train.py -- model_dir=$MODEL_DIR --dataset_name='newscommentary_paracrawl' \
  --batch_size=128 --num_train_steps=500000 \
  --emb_dim=512 --mlp_dim=2048 --num_heads=8 --paracrawl_size=4500000 \
  --vocab_path='tokenizer/sentencepiece_model' \
  --data_dir=$DATA_DIR --chkpts_to_keep=10 \
  --checkpoint_freq=50000 --eval_frequency=10000 \
  --save_checkpoints=True \
  --compute_bleu=False

# Compute DC scores
python clf_infer.py --save_dir=$SCORES_DIR --bert_base_dir=$BERT_BASE \
--bert_clf_dir=$BERT_TRAINED --$TARGET_TEXT --slice=0
# Repeat for slices 0-14. Can be run in parallel

# Finetune model and compute IS scores
python train.py -- model_dir=$FT_DIR --dataset_name='newscommentary_paracrawl' \
  --batch_size=128 --num_train_steps=4000 --restore_checkpoints \
  --emb_dim=512 --mlp_dim=2048 --num_heads=8 --paracrawl_size=4500000 \
  --vocab_path='tokenizer/sentencepiece_model' \
  --data_dir=$DATA_DIR --chkpts_to_keep=1 \
  --pretrained_model_dir=$MODEL_DIR \
  --checkpoint_freq=100 --eval_frequency=100 \
  --save_checkpoints=True \
  --compute_bleu=False --learning_rate=0.005


python compute_is.py --dataset_name='newscommentary_paracrawl' \
  --data_dir=$DATA_DIR --model_dir=$SCORES_DIR \
  --batch_size=1024  \
  --emb_dim=512 --mlp_dim=2048 --num_heads=8 --paracrawl_size=4500000 \
  --vocab_path='tokenizer/sentencepiece_model' \
  --restore_checkpoints --pretrained_model_dir=$MODEL_DIR  \
  --is_score_filename="CDS_scores_pretrain" --is_save_path=$SCORES_DIR


python compute_is.py --dataset_name='newscommentary_paracrawl' \
  --data_dir=$DATA_DIR --model_dir=$SCORES_DIR\
  --batch_size=1024 \
  --emb_dim=512 --mlp_dim=2048 --num_heads=8 --paracrawl_size=4500000 \
  --vocab_path='tokenizer/sentencepiece_model' \
  --restore_checkpoints --pretrained_model_dir=$FT_DIR  \
  --is_score_filename="CDS_scores_ft" \
  --is_save_path=$SCORES_DIR \
  --is_diff_name="CDS_scores.csv" \
  --base_log_loss_file=$SCORES_DIR+"CDS_scores_pretrain.txt"


# Data selection
python train.py -- model_dir=$MODEL_DIR --dataset_name='newscommentary_paracrawl' \
  --aux_eval_dataset='newscomment_eval_ft' --model_dir=$DATA_SELECT_DIR\
  --batch_size=128 --num_train_steps=15000 \
  --emb_dim=512 --mlp_dim=2048 --num_heads=8 --paracrawl_size=4500000 \
  --vocab_path 'tokenizer/sentencepiece_model' --restore_checkpoints \
  --data_dir=$DATA_DIR --chkpts_to_keep=1 \
  --checkpoint_freq=5000 --eval_frequency=100 \
  --pretrained_model_dir=$MODEL_DIR --save_checkpoints=False \
  --is_scores_path=$SCORES_DIR+"CDS_scores.csv" --data_selection_size=5e5 \
  --compute_bleu=False
