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
python train_online.py \
  --alsologtostderr --sub_dir=0 \
  --env_name=HalfCheetah-v2 \
  --eval_target=4000 \
  --agent_name=sac \
  --total_train_steps=500000 \
  --gin_bindings="train_eval_online.model_params=(((300, 300), (200, 200),), 2)" \
  --gin_bindings="train_eval_online.batch_size=256" \
  --gin_bindings="train_eval_online.optimizers=(('adam', 0.0005),)"
