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

python -m trainer.py \
  --t2t_usr_dir=$usr_dir \
  --model=sparse_transformer \
  --problem=translate_ende_wmt32k_packed \
  --hparams_set='sparse_transformer_variational_dropout_tpu' \
  --output_dir="/tmp/training/directory" \
  --worker_gpu=1 \
  --train_steps=500000 \
  --dkl_weight=8.89e-08 \
  --dkl_weight_start=0 \
  --dkl_weight_diff=200000 \
  --dkl_weight_fn="cubic" \
  --regularization="none" \
  --logtostderr
