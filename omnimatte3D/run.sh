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
python3 -m venv omnimatte3d
source ./omnimatte3d/bin/activate
pip install -r omnimatte3d/requirements.txt
# Run train.
python -m omnimatte3D.main \
  --workdir=/tmp/train_run \
  --is_train=True \
  --config.dataset.name=davis \
  --config.dataset.basedir=/path/to/datset/ \
  --config.dataset.scene=scene_name \
  --config.dataset.batch_size=4 \
  --config.model.name=ldi \
  --config.train.log_loss_every_steps=2000 \
  --config.train.checkpoint_every_steps=1000 \
  --config.train.max_steps=30000 \
  --config.train.switch_steps=100 \
  --config.train.scheduler=cosine \
  --config.train.lr_init=5.e-4 \
  --config.train.weight_decay=0.0000 \
  --config.train.crop_projection=False \
  --config.loss.disp_layer_alpha=1.0 \
  --config.loss.disp_smooth_alpha=0.5 \
  --config.loss.src_rgb_recon_alpha=1.0 \
  --config.loss.proj_far_rgb_alpha=1.0 \
  --config.loss.fg_mask_alpha=0.01 \
  --config.loss.shadow_smooth_alpha=0.01 \
  --config.loss.fg_alpha_reg_l0_alpha=0.001 \
  --config.loss.fg_alpha_reg_l1_alpha=0.0005
