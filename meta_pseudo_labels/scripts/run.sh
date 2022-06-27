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

source third_party/google_research/google_research/meta_pseudo_labels/scripts/imports.sh

ml_python \
  --adhoc_import_modules=${import_modules} \
  third_party/google_research/google_research/meta_pseudo_labels/main.py --xm_runlocal -- \
    --nouse_xla_sharding \
    --num_cores_per_replica=1 \
    --task_mode="train" \
    --dataset_name="${dataset_name}" \
    --output_dir="/usr/local/google/home/$USER/Desktop/dev_outputs/${dataset_name}" \
    --model_type=${model_type} \
    --label_smoothing=${label_smoothing} \
    --log_every=${log_every} \
    --master="${master}" \
    --train_batch_size=${train_batch_size} \
    --image_size=${image_size} \
    --num_classes=${num_classes} \
    --lr=${lr} \
    --ema_start=${ema_start} \
    --optim_type=${optim_type} \
    --lr_decay_type=${lr_decay_type} \
    --weight_decay=${weight_decay} \
    --dense_dropout_rate=${dense_dropout_rate} \
    --stochastic_depth_drop_rate=${stochastic_depth_drop_rate} \
    --num_shards_per_worker=1024 \
    --save_every=${save_every} \
    --use_bfloat16 \
    --use_tpu \
    --use_augment \
    --augment_magnitude=17 \
    --reset_output_dir \
    --eval_batch_size=64 \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=250 \
    --ema_decay=1.0 \
    --alsologtostderr \
    --running_local_dev \
    --uda_warmup_steps=1000 \
    --uda_data=${uda_data} \
    --uda_temp=0.75
    "$@"
