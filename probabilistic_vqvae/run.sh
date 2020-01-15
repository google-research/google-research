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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tf-nightly
pip install tfp-nightly
pip install -r probabilistic_vqvae/requirements.txt
python -m probabilistic_vqvae.mnist_experiments \
          --max_steps=500 \
          --latent_size 2 --num_codes 2 --code_size 4 --base_depth 2 --batch_size 4 \
          --nouse_autoregressive_prior
