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


DATA_DIR=$(pwd)/unit_test_data/

# First generate the data by running the following command.
python generate_datasets.py --data_dir="${DATA_DIR}" \
                            --class_ids="0,1" \
                            --max_data_seed=5

# Evaluating (only evolving the setup) a hand designed Neural Network on
# projected binary tasks. Utility script to check whether the tasks are
# ready.
bazel run -c opt \
  --copt=-DMAX_SCALAR_ADDRESSES=20 \
  --copt=-DMAX_VECTOR_ADDRESSES=20 \
  --copt=-DMAX_MATRIX_ADDRESSES=20 \
  :run_search_experiment -- \
  --search_experiment_spec=" \
    search_tasks { \
      tasks { \
        projected_binary_classification_task { \
          dataset_name: 'cifar10' \
          positive_class: 0 \
          negative_class: 1 \
          path: '${DATA_DIR}' \
          max_supported_data_seed: 2 \
        } \
        features_size: 16 \
        num_train_examples: 8000 \
        num_valid_examples: 1000 \
        num_train_epochs: 1 \
        num_tasks: 1 \
        eval_type: ACCURACY \
      } \
    } \
    setup_ops: [VECTOR_GAUSSIAN_SET_OP, MATRIX_GAUSSIAN_SET_OP] \
    predict_ops: [] \
    learn_ops: [] \
    setup_size_init: 2 \
    predict_size_init: 1 \
    learn_size_init: 4 \
    train_budget {train_budget_baseline: NEURAL_NET_ALGORITHM} \
    fitness_combination_mode: MEAN_FITNESS_COMBINATION \
    population_size: 10 \
    tournament_size: 10 \
    initial_population: NEURAL_NET_ALGORITHM \
    max_train_steps: 80000 \
    allowed_mutation_types {
      mutation_types: [ALTER_PARAM_MUTATION_TYPE, IDENTITY_MUTATION_TYPE] \
    } \
    mutate_prob: 1.0 \
    progress_every: 100 \
    " \
  --final_tasks="
    tasks {
        projected_binary_classification_task { \
          dataset_name: 'cifar10' \
          positive_class: 0 \
          negative_class: 1 \
          path: '${DATA_DIR}' \
        } \
      features_size: 16 \
      num_train_examples: 8000 \
      num_valid_examples: 1000 \
      num_train_epochs: 1 \
      data_seeds: 3 \
      num_tasks: 1 \
      eval_type: ACCURACY \
    } \
    " \
  --random_seed=1000060 \
  --select_tasks="
    tasks {
        projected_binary_classification_task { \
          dataset_name: 'cifar10' \
          positive_class: 0 \
          negative_class: 1 \
          path: '${DATA_DIR}' \
        } \
      features_size: 16 \
      num_train_examples: 8000 \
      num_valid_examples: 1000 \
      num_train_epochs: 1 \
      data_seeds: 2 \
      num_tasks: 1 \
      eval_type: ACCURACY \
    } \
    "
