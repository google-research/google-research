# Copyright 2020 The Google Research Authors.
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

virtualenv -p python3 .
source ./bin/activate

pip install -r hyperbolic_discount/requirements.txt

python -m hyperbolic_discount.train \
  --agent_name=hyperbolic_rainbow \
  --gin_files=hyperbolic_discount/configs/hyperbolic_rainbow_agent.gin \
  --base_dir=/tmp/base_dir \
  --gin_bindings=HyperRainbowAgent.gamma_max=0.99 \
  --gin_bindings=HyperRainbowAgent.number_of_gammas=10 \
  --gin_bindings=HyperRainbowAgent.hyp_exponent=0.01 \
  --schedule=continuous_train
