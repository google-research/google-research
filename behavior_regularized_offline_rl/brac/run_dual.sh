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
ALPHA=10.0
PLR=3e-6
VALUE_PENALTY=False
DIVERGENCE=kl
ENV=HalfCheetah-v2
DATA=example
python train_offline.py \
  --alsologtostderr --sub_dir=auto \
  --env_name=$ENV \
  --agent_name=brac_dual \
  --data_name=$DATA \
  --gin_bindings="brac_dual_agent.Agent.alpha=$ALPHA" \
  --gin_bindings="brac_dual_agent.Agent.train_alpha=False" \
  --gin_bindings="brac_dual_agent.Agent.value_penalty=$VALUE_PENALTY" \
  --gin_bindings="brac_dual_agent.Agent.target_divergence=0.0" \
  --gin_bindings="brac_dual_agent.Agent.train_alpha_entropy=False" \
  --gin_bindings="brac_dual_agent.Agent.alpha_entropy=0.0" \
  --gin_bindings="brac_dual_agent.Agent.divergence_name='$DIVERGENCE'" \
  --gin_bindings="brac_dual_agent.Agent.c_iter=3" \
  --gin_bindings="train_eval_offline.model_params=(((300, 300), (200, 200), (300, 300)), 2)" \
  --gin_bindings="train_eval_offline.batch_size=256" \
  --gin_bindings="train_eval_offline.optimizers=(('adam', 1e-3), ('adam', $PLR), ('adam', 1e-3), ('adam', 1e-3))"
