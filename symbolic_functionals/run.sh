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

virtualenv -p python3 .
source ./bin/activate

pip install -r symbolic_functionals/requirements.txt

python3 -m symbolic_functionals.syfes.example.single_run \
--config=symbolic_functionals/syfes/example/config.py \
--config.infra.cache_fingerprints=True \
--config.infra.history_data_name=/tmp/ \
--config.dataset.dataset_directory=/tmp/scf_wb97mv_def2qzvppd/ \
--config.dataset.mgcdb84_types=TCE \
--config.dataset.targets=b97x_u_short \
--config.dataset.num_targets=10 \
--config.xc.mutation_base=empty_functional \
--config.xc.mutation_base_losses=122.34,115.64,80.23 \
--config.xc.feature_names_x=u \
--config.xc.feature_names_css=u \
--config.xc.feature_names_cos=u \
--config.xc.component_mutation_probabilities=1.,0.,0. \
--config.xc.instruction_pool=/tmp/instruction_pool/b79x_u_minimum.json \
--config.xc.num_shared_parameters=3,0,0 \
--config.xc.num_variables=3 \
--config.xc.max_num_instructions=4 \
--config.opt.num_opt_trials=10
