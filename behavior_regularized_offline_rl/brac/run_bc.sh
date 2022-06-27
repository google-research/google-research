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
python train_offline \
  --alsologtostderr --sub_dir=0 \
  --env_name=HalfCheetah-v2 \
  --agent_name=bc \
  --data_name=example \
  --total_train_steps=300000 \
  --n_train=1000000 \
  --gin_bindings="train_eval_offline.model_params=((200, 200),)" \
  --gin_bindings="train_eval_offline.batch_size=256" \
  --gin_bindings="train_eval_offline.optimizers=(('adam', 5e-4),)"
