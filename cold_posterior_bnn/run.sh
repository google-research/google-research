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

# Runs unit tests and executes two epochs of sgmcmc sampling for the resnet and
# lstm model.
#
# Has to be executed from the parent folder by the shell command
# $ cold_posterior_bnn/run.sh


set -e
set -x

# setup virtual environment and install packages
virtualenv -p python3 .
source ./bin/activate

pip install -r cold_posterior_bnn/requirements.txt

# unit tests
python -m cold_posterior_bnn.core.ensemble_test
python -m cold_posterior_bnn.core.frn_test
python -m cold_posterior_bnn.core.model_test
python -m cold_posterior_bnn.core.prior_test
python -m cold_posterior_bnn.core.priorfactory_test
python -m cold_posterior_bnn.core.sgmcmc_test
python -m cold_posterior_bnn.core.statistics_test
python -m cold_posterior_bnn.core.diagnostics_test

# example of run with SG-MCMC, on CIFAR10 with ResNet
python -m cold_posterior_bnn.train \
  --model="resnet" \
  --dataset="cifar10" \
  --train_epochs=2 \
  --cycle_length=1 \
  --cycle_start_sampling=1 \
  --batch_size=512 \
  --pfac="gaussian"

# example of run with SG-MCMC, on IMDB with CNN-LSTM
python -m cold_posterior_bnn.train \
  --model="cnnlstm" \
  --dataset="imdb" \
  --train_epochs=2 \
  --cycle_length=1 \
  --cycle_start_sampling=1 \
  --batch_size=512 \
  --pfac="gaussian"
