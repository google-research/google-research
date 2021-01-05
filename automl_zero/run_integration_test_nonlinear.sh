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

# TODO(ereal): fix mutate_*_size_* values. Affects random number generation.

# Nonlinear example with Neural Network as init model:
bazel run -c opt \
  --copt=-DMAX_SCALAR_ADDRESSES=5 \
  --copt=-DMAX_VECTOR_ADDRESSES=9 \
  --copt=-DMAX_MATRIX_ADDRESSES=2 \
  :run_search_experiment -- \
  --search_experiment_spec=" \
    search_tasks { \
      tasks { \
        scalar_2layer_nn_regression_task {} \
        features_size: 4 \
        num_train_examples: 1000 \
        num_valid_examples: 100 \
        num_tasks: 1 \
        eval_type: RMS_ERROR \
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
    population_size: 1000 \
    tournament_size: 10 \
    initial_population: INTEGRATION_TEST_DAMAGED_NEURAL_NET_ALGORITHM \
    max_train_steps: 100000000 \
    allowed_mutation_types {
      mutation_types: [ALTER_PARAM_MUTATION_TYPE, INSERT_INSTRUCTION_MUTATION_TYPE, REMOVE_INSTRUCTION_MUTATION_TYPE, TRADE_INSTRUCTION_MUTATION_TYPE] \
    } \
    mutate_prob: 1.0 \
    mutate_setup_size_min: 6 \
    mutate_setup_size_max: 7 \
    mutate_predict_size_min: 3 \
    mutate_predict_size_max: 4 \
    mutate_learn_size_min: 9 \
    mutate_learn_size_max: 10 \
    progress_every: 10000 \
    " \
  --final_tasks=" \
    tasks { \
      scalar_2layer_nn_regression_task {} \
      features_size: 4 \
      num_train_examples: 1000 \
      num_valid_examples: 100 \
      num_tasks: 1 \
      eval_type: RMS_ERROR \
      param_seeds: [9000] \
      data_seeds: [19000] \
    } \
    " \
  --random_seed=100001 \
  --select_tasks=" \
    tasks { \
      scalar_2layer_nn_regression_task {} \
      features_size: 4 \
      num_train_examples: 1000 \
      num_valid_examples: 100 \
      num_tasks: 1 eval_type: \
      RMS_ERROR param_seeds: [9000] \
      data_seeds: [19000] \
    } \
    "
