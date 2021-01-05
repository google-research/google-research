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

# Runs the experiment that reproduces figure 1 and figure 2 in the paper Wenzel
# et al. 2020, "How Good is the Bayes Posterior in Deep Neural Networks Really?".
# The individual runs are executed in sequential order which takes around 30
# days on the GPU Geforce RTX 2080 Ti. For faster execution run in parallel or
# run the simplified version of this experiment ("run_resnet_experiment_small.sh")
# which produces an approximate version of figure 1 and 2 and only takes roughly
# 2 days on a single GPU.
#
# Has to be executed from the parent folder by the shell command
# $ cold_posterior_bnn/run_resnet_experiment.sh


# setup virtual environment and install packages
virtualenv -p python3 .
source ./bin/activate

pip install -r cold_posterior_bnn/requirements.txt

# Output directory for the results of experiments, string should end with '/'
output_dir='cold_posterior_bnn/results_resnet/'

# Exeriment settings
train_epochs=1500
init_learning_rate=0.1
dataset='cifar10'
model='resnet'
method='sgmcmc'
momentum_decay=0.98
batch_size=128
cycle_start_sampling=150
cycle_length=50

# Hyperparameters to sweep
num_runs=3  # Number of repeated runs per hyperparameter setting
min_log_temp=-4  # Minimal log_10 temperature of sweep
max_log_temp=0  # Maximal log_10 temperature of sweep
step_temp=0.25  # step between temperatures in log_10 space

# Generate parameter lists to sweep
seed_range=($(seq 1 $num_runs))
temp_range=($(seq $min_log_temp $step_temp $max_log_temp))

# Run experiment binary
experiment_id=-1
for seed in ${seed_range[@]}; do
  for log_temperature in ${temp_range[@]}; do
    experiment_id=$((experiment_id+1))
    # map temperature to log scale
    temperature=$(awk -v var=$log_temperature 'BEGIN{print 10^(var)}')
    # run experiment
    printf "\n\n* * * Run sgmcmc for seed = $seed, temperature = $temperature. Experiment ID = $experiment_id. * * *\n\n\n"
    python -m cold_posterior_bnn.train \
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
