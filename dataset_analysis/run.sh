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

virtualenv -p python3 .
source ./bin/activate
pip install numpy
pip install six
pip install -r dataset_analysis/requirements.txt

echo "Performing training & evaluation..."
python -m dataset_analysis.bert_classifier \
--do_train=true \
--train_fname=train.tsv \
--dev_fname=dev.tsv \
--multilabel=true \
--data_dir=data/model_input \
--target_file=data/targets.txt \
--vocab_file=pretrained_BERT/cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_BERT/cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_BERT/cased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=50 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=4 \
--sentiment=1 \
--entailment=1e-3 \
--correlation=10 \
--save_checkpoints_steps=1361 \
--output_dir=checkpoints \
--do_lower_case=false \
--alsologtostderr \
--minloglevel=0

echo "Performing prediction..."
python -m dataset_analysis.bert_classifier \
--do_train=false \
--do_predict=true \
--calculate_metrics=true \
--test_fname=test.tsv \
--multilabel=true \
--data_dir=data/model_input \
--target_file=data/targets.txt \
--vocab_file=pretrained_BERT/cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_BERT/cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_BERT/cased_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=50 \
--output_dir=checkpoints \
--do_lower_case=false \
--alsologtostderr \
--minloglevel=0


