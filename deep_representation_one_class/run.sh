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

pip install -r deep_representation_one_class/requirements.txt

# An example run for cifarood.
python -m deep_representation_one_class.train_and_eval_loop \
  --model_dir=/tmp/model \
  --method=UnsupEmbed \
  --dataset=cifar10ood \
  --net_type=ResNet18 \
  --net_width=1 \
  --latent_dim=4 \
  --aug_list=cnr0.5+hflip+jitter_b0.4_c0.4_s0.4_h0.4+gray0.2+blur0.5,+rotate90,+rotate180,+rotate270 \
  --aug_list_for_test=x,+rotate90,+rotate180,+rotate270 \
  --input_shape=32,32,3 \
  --sched_type=cos \
  --sched_freq=epoch \
  --learning_rate=0.01 \
  --momentum=0.9 \
  --weight_decay=0.0003 \
  --nesterov=false \
  --regularize_bn=false \
  --head_dims=512,512,512,512,512,512,512,512 \
  --file_path=/tmp/model \
  --num_batch=0 \
  --num_epoch=2048 \
  --category=0 \
  --batch_size=64
