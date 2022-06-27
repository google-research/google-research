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

DATA_DIR=$(pwd)/binary_cifar10_data/

# Evaluating (only evolving the setup) a hand designed Neural Network on
# projected binary tasks. Utility script to check whether the tasks are
# ready.
bazel run -c opt \
  --copt=-DMAX_SCALAR_ADDRESSES=5 \
  --copt=-DMAX_VECTOR_ADDRESSES=9 \
  --copt=-DMAX_MATRIX_ADDRESSES=2 \
  :run_search_experiment -- \
  --search_experiment_spec=" \
    search_tasks { \
      tasks { \
        projected_binary_classification_task { \
          dataset_name: 'cifar10' \
          path: '${DATA_DIR}' \
          held_out_pairs {positive_class: 0 negative_class: 5} \
          held_out_pairs {positive_class: 0 negative_class: 9} \
          held_out_pairs {positive_class: 1 negative_class: 8} \
          held_out_pairs {positive_class: 2 negative_class: 9} \
          held_out_pairs {positive_class: 3 negative_class: 5} \
          held_out_pairs {positive_class: 3 negative_class: 6} \
          held_out_pairs {positive_class: 3 negative_class: 8} \
          held_out_pairs {positive_class: 4 negative_class: 6} \
          held_out_pairs {positive_class: 8 negative_class: 9} \
          max_supported_data_seed: 100 \
        } \
        features_size: 16 \
        num_train_examples: 8000 \
        num_valid_examples: 1000 \
        num_train_epochs: 1 \
        num_tasks: 10 \
        eval_type: ACCURACY \
      } \
    } \
    setup_ops: [SCALAR_CONST_SET_OP, SCALAR_GAUSSIAN_SET_OP, SCALAR_UNIFORM_SET_OP, VECTOR_GAUSSIAN_SET_OP, VECTOR_UNIFORM_SET_OP, MATRIX_GAUSSIAN_SET_OP, MATRIX_UNIFORM_SET_OP] \
    predict_ops: [SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MAX_OP, VECTOR_INNER_PRODUCT_OP, VECTOR_SUM_OP] \
    learn_ops: [SCALAR_SUM_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_VECTOR_PRODUCT_OP, VECTOR_SUM_OP, VECTOR_HEAVYSIDE_OP, VECTOR_PRODUCT_OP, VECTOR_OUTER_PRODUCT_OP, MATRIX_SUM_OP] \
    setup_size_init: 1 \
    mutate_setup_size_min: 1 \
    mutate_setup_size_max: 7 \
    predict_size_init: 1 \
    mutate_predict_size_min: 1 \
    mutate_predict_size_max: 11 \
    learn_size_init: 1 \
    mutate_learn_size_min: 1 \
    mutate_learn_size_max: 23 \
    train_budget {train_budget_baseline: NEURAL_NET_ALGORITHM} \
    fitness_combination_mode: MEAN_FITNESS_COMBINATION \
    population_size: 100 \
    tournament_size: 10 \
    initial_population: NO_OP_ALGORITHM \
    max_train_steps: 100000000000 \
    allowed_mutation_types {
      mutation_types: [ALTER_PARAM_MUTATION_TYPE, RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE, INSERT_INSTRUCTION_MUTATION_TYPE, REMOVE_INSTRUCTION_MUTATION_TYPE] \
    } \
    mutate_prob: 0.9 \
    progress_every: 10000 \
    " \
  --final_tasks="
    tasks { \
      projected_binary_classification_task { \
        dataset_name: 'cifar10' \
        path: '${DATA_DIR}' \
        held_out_pairs {positive_class: 0 negative_class: 1} \
        held_out_pairs {positive_class: 0 negative_class: 2} \
        held_out_pairs {positive_class: 0 negative_class: 3} \
        held_out_pairs {positive_class: 0 negative_class: 4} \
        held_out_pairs {positive_class: 0 negative_class: 6} \
        held_out_pairs {positive_class: 0 negative_class: 7} \
        held_out_pairs {positive_class: 0 negative_class: 8} \
        held_out_pairs {positive_class: 1 negative_class: 2} \
        held_out_pairs {positive_class: 1 negative_class: 3} \
        held_out_pairs {positive_class: 1 negative_class: 4} \
        held_out_pairs {positive_class: 1 negative_class: 5} \
        held_out_pairs {positive_class: 1 negative_class: 6} \
        held_out_pairs {positive_class: 1 negative_class: 7} \
        held_out_pairs {positive_class: 1 negative_class: 9} \
        held_out_pairs {positive_class: 2 negative_class: 3} \
        held_out_pairs {positive_class: 2 negative_class: 4} \
        held_out_pairs {positive_class: 2 negative_class: 5} \
        held_out_pairs {positive_class: 2 negative_class: 6} \
        held_out_pairs {positive_class: 2 negative_class: 7} \
        held_out_pairs {positive_class: 2 negative_class: 8} \
        held_out_pairs {positive_class: 3 negative_class: 4} \
        held_out_pairs {positive_class: 3 negative_class: 7} \
        held_out_pairs {positive_class: 3 negative_class: 9} \
        held_out_pairs {positive_class: 4 negative_class: 5} \
        held_out_pairs {positive_class: 4 negative_class: 7} \
        held_out_pairs {positive_class: 4 negative_class: 8} \
        held_out_pairs {positive_class: 4 negative_class: 9} \
        held_out_pairs {positive_class: 5 negative_class: 6} \
        held_out_pairs {positive_class: 5 negative_class: 7} \
        held_out_pairs {positive_class: 5 negative_class: 8} \
        held_out_pairs {positive_class: 5 negative_class: 9} \
        held_out_pairs {positive_class: 6 negative_class: 7} \
        held_out_pairs {positive_class: 6 negative_class: 8} \
        held_out_pairs {positive_class: 6 negative_class: 9} \
        held_out_pairs {positive_class: 7 negative_class: 8} \
        held_out_pairs {positive_class: 7 negative_class: 9} \
        max_supported_data_seed: 100 \
      } \
      features_size: 16 \
      num_train_examples: 8000 \
      num_valid_examples: 1000 \
      num_train_epochs: 1 \
      num_tasks: 100 \
      eval_type: ACCURACY \
    } \
    " \
  --random_seed=1000060 \
  --select_tasks="
    tasks { \
      projected_binary_classification_task { \
        dataset_name: 'cifar10' \
        path: '${DATA_DIR}' \
        held_out_pairs {positive_class: 0 negative_class: 5} \
        held_out_pairs {positive_class: 0 negative_class: 9} \
        held_out_pairs {positive_class: 1 negative_class: 8} \
        held_out_pairs {positive_class: 2 negative_class: 9} \
        held_out_pairs {positive_class: 3 negative_class: 5} \
        held_out_pairs {positive_class: 3 negative_class: 6} \
        held_out_pairs {positive_class: 3 negative_class: 8} \
        held_out_pairs {positive_class: 4 negative_class: 6} \
        held_out_pairs {positive_class: 8 negative_class: 9} \
        max_supported_data_seed: 100 \
      } \
      features_size: 16 \
      num_train_examples: 8000 \
      num_valid_examples: 1000 \
      num_train_epochs: 1 \
      num_tasks: 10 \
      eval_type: ACCURACY \
    } \
    "
