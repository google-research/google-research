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

# Linear example with early stopping:
bazel run -c opt \
  --copt=-DMAX_SCALAR_ADDRESSES=4 \
  --copt=-DMAX_VECTOR_ADDRESSES=3 \
  --copt=-DMAX_MATRIX_ADDRESSES=1 \
  :run_search_experiment -- \
  --search_experiment_spec=" \
    search_tasks { \
      tasks { \
        scalar_linear_regression_task {} \
        features_size: 4 \
        num_train_examples: 100 \
        num_valid_examples: 100 \
        num_tasks: 10 \
        eval_type: RMS_ERROR \
      } \
    } \
    setup_ops: [SCALAR_CONST_SET_OP] \
    predict_ops: [VECTOR_INNER_PRODUCT_OP] \
    learn_ops: [SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_VECTOR_PRODUCT_OP, VECTOR_SUM_OP] \
    setup_size_init: 5 \
    predict_size_init: 1 \
    learn_size_init: 4 \
    fec {num_train_examples: 10 num_valid_examples: 10} \
    fitness_combination_mode: MEAN_FITNESS_COMBINATION \
    population_size: 1000 \
    tournament_size: 10 \
    initial_population: RANDOM_ALGORITHM \
    max_train_steps: 100000000 \
    allowed_mutation_types {
      mutation_types: [ALTER_PARAM_MUTATION_TYPE, RANDOMIZE_INSTRUCTION_MUTATION_TYPE, RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE] \
    } \
    mutate_prob: 0.9 \
    progress_every: 10000 \
    " \
  --final_tasks=" \
    tasks { \
      scalar_linear_regression_task {} \
      features_size: 4 \
      num_train_examples: 10000 \
      num_valid_examples: 100 \
      num_tasks: 100 \
      eval_type: RMS_ERROR \
    } \
    " \
  --select_tasks=" \
    tasks { \
      scalar_linear_regression_task {} \
      features_size: 4 \
      num_train_examples: 10000 \
      num_valid_examples: 100 \
      num_tasks: 100 \
      eval_type: \
      RMS_ERROR \
    } \
    " \
  --random_seed=100021

