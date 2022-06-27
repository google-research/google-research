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
set -e
set -x

conda create -n vae_ood python=3.7.10
source activate vae_ood

pip install -r vae_ood/requirements.txt
python -m vae_ood.main --dataset fashion_mnist --do_train --do_eval --latent_dim 20 --experiment_dir vae_ood/models/cont_bernoulli/
