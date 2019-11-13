# Copyright 2019 The Google Research Authors.
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
# Runs SCAE on 40x40 MNIST where part templates and their mixing probabilisties
# are learned separately.
set -e
set -x

source stacked_capsule_autoencoders/setup_virtualenv.sh
python -m stacked_capsule_autoencoders.train\
  --name=mnist\
  --model=scae\
  --dataset=mnist\
  --max_train_steps=300000\
  --batch_size=128\
  --lr=3e-5\
  --use_lr_schedule=True\
  --canvas_size=40\
  --n_part_caps=40\
  --n_obj_caps=32\
  --colorize_templates=True\
  --use_alpha_channel=True\
  --posterior_between_example_sparsity_weight=0.2\
  --posterior_within_example_sparsity_weight=0.7\
  --prior_between_example_sparsity_weight=0.35\
  --prior_within_example_constant=4.3\
  --prior_within_example_sparsity_weight=2.\
  --color_nonlin='sigmoid'\
  --template_nonlin='sigmoid'\
  "$@"
