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

# Runs a faster simplified version of the experiment defined in the file
# "run_resnet_experiment.sh". This experiment takes roughly 2 days on the GPU
# Geforce RTX 2080 Ti, in contrast to 30 days for the full experiment.
# This file approximately reproduces figure 1 and figure 2 in the paper Wenzel
# et al. 2020, "How Good is the Bayes Posterior in Deep Neural Networks Really?".
# The results can be found in the folder "cold_posterior_bnn/results_resnet/".
#
# Has to be executed from the parent folder by the shell command
# $ cold_posterior_bnn/run_resnet_experiment_small.sh


# setup virtual environment and install packages
virtualenv -p python3 .
source ./bin/activate

pip install -r cold_posterior_bnn/requirements.txt

# Output directory for the results of experiments, string should end with '/'
output_dir='cold_posterior_bnn/results_resnet/'

# Exeriment settings
train_epochs=750
init_learning_rate=0.1
dataset='cifar10'
model='resnet'
method='sgmcmc'
momentum_decay=0.98
batch_size=128
cycle_start_sampling=150
cycle_length=50

# Hyperparameters to sweep
num_runs=1  # Number of repeated runs per hyperparameter setting

# Generate parameter lists to sweep
seed_range=($(seq 1 $num_runs))
temp_range=(-4.0 -3.0 -2.0 -1.0 -0.75 -0.5 -0.25 0.0)

# Run experiment binary
experiment_id=-1
for seed in ${seed_range[@]}; do
  for log_temperature in ${temp_range[@]}; do
    experiment_id=$((experiment_id+1))
    # map temperature to log scale
    temperature=$(awk -v var=$log_temperature 'BEGIN{print 10^(var)}')
    # run experiment
    printf "\n\n* * * Run sgmcmc for seed = $seed, temperature = $temperature. Experiment ID = $experiment_id. * * *\n\n\n"
    python -m cold_posterior_bnn.run_sgmcmc \
      --model=$model \
      --dataset=$dataset \
      --train_epochs=$train_epochs \
      --cycle_length=$cycle_length \
      --cycle_start_sampling=$cycle_start_sampling \
      --batch_size=$batch_size \
      --pfac="gaussian" \
      --temperature=$temperature \
      --seed=$seed \
      --init_learning_rate=$init_learning_rate \
      --output_dir="${output_dir}run_${experiment_id}"  \
      --experiment_id=$experiment_id  \
      --write_experiment_metadata_to_csv=True
  done
done

printf "\n\n* * * All runs finished. * * *\n"
