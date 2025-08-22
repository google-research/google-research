#!/bin/bash
# Copyright 2025 The Google Research Authors.
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
set -x

pip install -r universal_embedding_challenge/requirements.txt

rm -rf /tmp/models
git clone https://github.com/tensorflow/models.git /tmp/models
export PYTHONPATH=$PYTHONPATH:/tmp/models
pip install --user -r /tmp/models/official/requirements.txt

python -m universal_embedding_challenge.train \
  --experiment=vit_with_bottleneck_imagenet_pretrain \
  --mode=train_and_eval \
  --model_dir=/tmp/imagenet1k_test \
  --params_override="{'task': {'model': {'backbone': {'vit': {'model_name': 'vit-ti16'}}}, 'train_data': {'global_batch_size': 1}, 'validation_data': {'global_batch_size': 1}}}"

pytest -v uec.metrics_test.py

pytest -v uec.read_retrieval_solution_test.py