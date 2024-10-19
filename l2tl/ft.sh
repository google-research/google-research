# Copyright 2024 The Google Research Authors.
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

model_name=finetune_svhn
steps=1200

python3 finetuning.py \
    --target_dataset=svhn_cropped_small \
    --train_steps=$steps \
    --model_dir=trained_models/${model_name} \
    --train_batch_size=8 \
    --src_num_classes=5 \
    --target_base_learning_rate=0.005 \
    --warm_start_ckpt_path=trained_models/mnist_pretrain/model.ckpt-2000

python3 evaluate.py \
    --ckpt_path=trained_models/${model_name}/model.ckpt-$steps \
    --src_num_classes=5 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_dense_dst
