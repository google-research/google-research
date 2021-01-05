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

model_name=l2tl_svhn
steps=1200

python3 train_l2tl.py \
    --train_batch_size=8 \
    --learning_rate=0.005 \
    --rl_learning_rate=0.01 \
    --target_num_classes=5 \
    --train_steps=$steps \
    --source_train_batch_multiplier=2 \
    --loss_weight_scale=100. \
    --num_choices=100 \
    --first_pretrain_steps=0 \
    --target_val_batch_multiplier=4 \
    --target_train_batch_multiplier=1 \
    --model_dir=trained_models/${model_name} \
    --warm_start_ckpt_path=trained_models/mnist_pretrain/model.ckpt-2000

python3 evaluate.py \
    --ckpt_path=trained_models/${model_name}/model.ckpt-$steps \
    --target_dataset=svhn_cropped_small \
    --src_num_classes=5 \
    --cls_dense_name=final_target_dense
