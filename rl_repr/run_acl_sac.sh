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

python3 -m rl_repr.batch_rl.train_eval_offline \
  --downstream_mode online \
  --algo_name sac \
  --state_mask_value zero \
  --state_mask_dims 1 \
  --state_mask_eval 1 \
  --state_mask_index 1 \
  --task_name ant-expert-v0 \
  --embed_learner acl \
  --state_embed_dim 256 \
  --embed_training_window 8 \
  --embed_pretraining_steps 200_000 \
  --alsologtostderr
