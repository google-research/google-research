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

# CUDA
export CUDA_VISIBLE_DEVICES=""

# for imports
root_dir=""
import_modules="${root_dir}.common_utils"
import_modules="${import_modules},${root_dir}.augment"
import_modules="${import_modules},${root_dir}.data_utils"
import_modules="${import_modules},${root_dir}.flag_utils"
import_modules="${import_modules},${root_dir}.modeling"
import_modules="${import_modules},${root_dir}.modeling_utils"
import_modules="${import_modules},${root_dir}.training_utils"

export import_modules=${import_modules}

# Master job bns address
master="/bns/ym/borg/ym/bns/hyhieu/hyhieu_headless_jf_2x2_19331757.1.tpu_worker"

stochastic_depth_drop_rate=0.2
dense_dropout_rate=0.2
lr=0.1
optim_type="momentum"
lr_decay_type="cosine"
ema_start=0
num_train_steps=500
uda_data=1

# dataset
dataset_name="cifar10_4000_mpl"
# dataset_name="imagenet_10_percent_mpl"

if [ ${dataset_name} = "cifar10_4000_mpl" ]; then
  optim_type='momentum'
  lr=0.1
  log_every=50
  image_size=32
  model_type="wrn-28-2"
  num_classes=10
  save_every=1000
  num_train_steps=5000
  weight_decay=5e-4
  train_batch_size=64
  label_smoothing=0.
  uda_data=3
  dense_dropout_rate=0.1
elif [ ${dataset_name} = "imagenet_10_percent_mpl" ]; then
  num_train_steps=1000
  save_every=250
  log_every=100
  image_size=224
  model_type='resnet-50'
  dense_dropout_rate=0.1
  weight_decay=1e-4
  num_classes=1000
  train_batch_size=16
  label_smoothing=0.1
  uda_data=2
else
  echo "Undefined dataset_name ${dataset_name}"
  exit
fi
