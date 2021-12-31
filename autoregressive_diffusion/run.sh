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

# Runs the main image experiment.
python -m autoregressive_diffusion.experiments.images.main --work_unit_dir=results/images --config autoregressive_diffusion/experiments/images/config.py \
  --config.num_epochs 1 --config.architecture.n_channels 64 --config.architecture.num_res_blocks 0

# Runs the main language experiment.
python -m autoregressive_diffusion.experiments.language.main --work_unit_dir=results/language --config autoregressive_diffusion/experiments/language/configs/default.py \
  --config.num_train_steps 1 --config.num_layers 0

# Runs the main audio experiment.
python -m autoregressive_diffusion.experiments.audio.main --work_unit_dir=results/audio --config autoregressive_diffusion/experiments/audio/configs/sc09.py \
  --config.num_train_steps 1 --config.arch.config.num_blocks 1 --executable_name train_and_evaluate


