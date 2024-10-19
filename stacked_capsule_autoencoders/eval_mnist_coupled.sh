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

#!/bin/bash
set -e
set -x

source stacked_capsule_autoencoders/setup_virtualenv.sh
python -m stacked_capsule_autoencoders.eval_mnist_model\
  --snapshot=stacked_capsule_autoencoders/checkpoints/mnist_coupled/model.ckpt-300001\
  --canvas_size=40\
  --n_part_caps=24\
  --n_obj_caps=24\
  --colorize_templates=True\
  --use_alpha_channel=False\
  "$@"
