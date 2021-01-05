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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r learn_to_infer/requirements.txt

python3 -m learn_to_infer.run_gmm \
  --model_name="mean_scale_weight" \
  --num_encoders=1 \
  --num_decoders=1 \
  --num_heads=2 \
  --batch_size=4 \
  --eval_batch_size=4 \
  --min_k=3 \
  --max_k=3 \
  --num_steps=3 \
  --summarize_every=1

python3 -m learn_to_infer.run_ring \
  --num_encoders=1 \
  --num_decoders=1 \
  --num_heads=2 \
  --batch_size=4 \
  --eval_batch_size=4 \
  --k=3 \
  --num_steps=3 \
  --summarize_every=1

python3 -m learn_to_infer.run_lda \
  --model="topic_word" \
  --num_encoders=1 \
  --num_decoders=1 \
  --num_heads=2 \
  --key_dim=8 \
  --value_dim_per_head=4 \
  --embedding_dim=8 \
  --num_docs=10 \
  --num_topics=2 \
  --vocab_size=25 \
  --doc_length=5 \
  --batch_size=4 \
  --eval_batch_size=4 \
  --num_steps=3 \
  --summarize_every=1

python3 -m learn_to_infer.run_lda \
  --model="doc_topic" \
  --num_encoders=1 \
  --num_decoders=1 \
  --num_heads=2 \
  --key_dim=8 \
  --value_dim_per_head=4 \
  --embedding_dim=8 \
  --num_docs=10 \
  --num_topics=2 \
  --vocab_size=25 \
  --doc_length=5 \
  --batch_size=4 \
  --eval_batch_size=4 \
  --num_steps=3 \
  --summarize_every=1
