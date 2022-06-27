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

#! /bin/bash
set -e
set -x
virtualenv -p python3 .
source ./bin/activate

pip3 install tensorflow torch jax jaxlib
pip3 install git+https://github.com/google-research/flax.git@prerelease

cd opt_list/

python3 -m opt_list.examples.tf_v1
python3 -m opt_list.examples.tf_keras
python3 -m opt_list.examples.torch
python3 -m opt_list.examples.jax_flax
python3 -m opt_list.examples.jax_optimizers
python3 -m opt_list.examples.jax_optix
