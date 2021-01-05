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
#
# Run it from google-research/ directory
#

set -e
DIR=$HOME/synthetic_recursive
mkdir -p $DIR

python -m recursive_optimizer.synthetic_experiment \
--optimizer=adagrad \
--steps=200000 \
--learning_rate=0.01 \
--conditioning=min \
--inner_optimizer=SCINOL \
--eta=1000000.0 \
--tau=0.0 \
--betting_domain=0.5 \
--epsilon=1.0 \
--epsilon_v=1.0 \
--g_max=0.0 \
--distance=10 \
--skewness=750 \
--loss=abs \
--data_output=$DIR/adagrad_min_new_10distance_0.01lr_750sk.csv

python -m recursive_optimizer.synthetic_experiment \
--optimizer=recursive \
--steps=200000 \
--learning_rate=1.0 \
--conditioning=min \
--inner_optimizer=SCINOL \
--eta=1.0 \
--tau=0.0 \
--betting_domain=0.5 \
--epsilon=1.0 \
--epsilon_v=1.0 \
--g_max=0.0 \
--distance=10 \
--skewness=750 \
--loss=abs \
--data_output=$DIR/recursive_min_new_10distance_1.0eta_1.0lr_750sk.csv
