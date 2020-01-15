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
PHI=0.15
PLR=3e-05
ENV=HalfCheetah-v2
DATA=example
python train_offline.py \
  --alsologtostderr --sub_dir=auto \
  --env_name=$ENV \
  --agent_name=bcq \
  --data_name=$DATA \
  --total_train_steps=500000 \
  --gin_bindings="train_eval_offline.model_params=(((300, 300), (300, 300), (750, 750)), 2, $PHI)" \
  --gin_bindings="train_eval_offline.batch_size=100" \
  --gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3))"
