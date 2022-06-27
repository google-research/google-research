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
# bash drfact.sh train
# bash drfact.sh eval
CORPUS_PATH=/path/to/knowledge_corpus/
BERT_PATH=/path/to/bert_large/
INDEX_PATH=/path/to/local_drfact_index
MODEL_OUTPUT_DIR=/path/to/model_output_dir/

if [ "$1" = "train" ]; then
  echo "training mode"
  rm -r ${MODEL_OUTPUT_DIR}
  DO="do_train"
else
  echo "prediction mode"
  DO="do_predict --use_best_ckpt_for_predict --model_ckpt_toload model.ckpt-25"
fi

python run_drfact.py \
  --vocab_file ${BERT_PATH}/vocab.txt \
  --tokenizer_model_file None \
  --bert_config_file ${BERT_PATH}/bert_config.json \
  --tokenizer_type bert_tokenization \
  --output_dir ${MODEL_OUTPUT_DIR} \
  --train_file ${INDEX_PATH}/drfact_output_bert200/linked_arc_easy_train.jsonl \
  --predict_file ${INDEX_PATH}/drfact_output_bert200/linked_arc_easy_dev.jsonl \
  --predict_prefix dev \
  --test_file ${INDEX_PATH}/drfact_output_bert200/linked_arc_easy_dev.jsonl \
  --init_checkpoint ${BERT_PATH}/bert_model.ckpt \
  --train_data_dir ${INDEX_PATH}/drfact_output_bert200 \
  --test_data_dir ${INDEX_PATH}/drfact_output_bert200 \
  --f2f_index_dir ${INDEX_PATH}/fact2fact_index \
  --learning_rate 3e-05 \
  --train_batch_size 3 \
  --predict_batch_size 3 \
  --save_checkpoints_steps 25 \
  --iterations_per_loop 300 \
  --num_train_epochs 5.0 \
  --max_query_length 64 \
  --max_entity_len 5 \
  --qry_layers_to_use -1 \
  --qry_aggregation_fn concat \
  --question_dropout 0.2 \
  --question_num_layers 5 \
  --projection_dim 200 \
  --train_with_sparse  \
  --fix_sparse_to_one  \
  --predict_with_sparse  \
  --data_type hotpotqa \
  --model_type drfact \
  --supervision entity \
  --num_mips_neighbors 100 \
  --entity_score_aggregation_fn sum \
  --entity_score_threshold 1e-3 \
  --fact_score_threshold 1e-2 \
  --softmax_temperature 5.0 \
  --sparse_reduce_fn max \
  --sparse_strategy sparse_first \
  --num_hops 2 \
  --num_preds -1 \
  --embed_index_prefix bert_large \
  --alsologtostderr \
  --$DO

# tensorboard --logdir ${MODEL_OUTPUT_DIR}


