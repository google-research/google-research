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
# Contrastive learning with distribution augmentation
# DATA in {cifar10ood,cifar20ood,fmnistood,dogvscatood,celeba}
# CATEGORY in
#   {0..9..1} for cifar10ood
#   {0..19..1} for cifar20ood
#   {0..9..1} for fmnistood
#   {0..1..1} for dogvscatood, 0 for cat, 1 for dog as inlier
# Reproducing results in Table 2: 92.5 +- 0.6 on cifar10ood with OC-SVM
DATA=cifar10ood
METHOD=Contrastive
SEED=1
CATEGORY=0
MODEL_DIR='.' # [/path/to/directory/to/save/model]
python -m deep_representation_one_class.script.train_and_eval_loop.py \
  --model_dir="${MODEL_DIR}" \
  --method=${METHOD} \
  --file_path="${DATA}_${PREFIX}_s${SEED}_c${CATEGORY}" \
  --dataset=${DATA} \
  --category=${CATEGORY} \
  --seed=${SEED};
  --root='' \
  --net_type=ResNet18 \
  --net_width=1 \
  --latent_dim=0 \
  --aug_list="cnr0.5+hflip+jitter_b0.4_c0.4_s0.4_h0.4+gray0.2+blur0.5,+" \
  --aug_list_for_test="x" \
  --input_shape="32,32,3" \
  --optim_type=sgd \
  --sched_type=cos \
  --learning_rate=0.01 \
  --momentum=0.9 \
  --weight_decay=0.0003 \
  --head_dims="512,512,512,512,512,512,512,512,128" \
  --num_epoch=2048 \
  --batch_size=32 \
  --temperature=0.2 \
  --distaug_type rot

# To reproduce results in Figure 7,
#DATA=cifar10ood
#METHOD=Contrastive
#SEED=1
#CATEGORY=0
#MODEL_DIR='.' # [/path/to/directory/to/save/model]
#DISTAUG_TYPE in {1,2,3,4,5,6,7,8}
#python train_and_eval_loop.py \
#  --model_dir="${MODEL_DIR}" \
#  --method=${METHOD} \
#  --file_path="${DATA}_${PREFIX}_s${SEED}_c${CATEGORY}" \
#  --dataset=${DATA} \
#  --category=${CATEGORY} \
#  --seed=${SEED};
#  --root='' \
#  --net_type=ResNet18 \
#  --net_width=1 \
#  --latent_dim=0 \
#  --aug_list="cnr0.5+jitter_b0.4_c0.4_s0.4_h0.4+gray0.2+blur0.5,+" \
#  --aug_list_for_test="x" \
#  --input_shape="32,32,3" \
#  --optim_type=sgd \
#  --sched_type=cos \
#  --learning_rate=0.01 \
#  --momentum=0.9 \
#  --weight_decay=0.0003 \
#  --head_dims="512,512,512,512,512,512,512,512,128" \
#  --num_epoch=2048 \
#  --batch_size=32 \
#  --temperature=0.2 \
#  --distaug_type "${DISTAUG_TYPE}"
