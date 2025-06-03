#!/bin/bash
# Copyright 2025 The Google Research Authors.
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


# This script runs the BUSTLE synthesizer on all benchmark tasks.

set -e

DIR=$(dirname "$(readlink -f "$0")")
cd "${DIR}"

MODELS_DIR=${DIR}/models
RESULTS_DIR=${DIR}/results

mkdir -p ${RESULTS_DIR}

BUSTLE_MODEL=${MODELS_DIR}/"1000_epochs__64_64__e2__1e-5"

TIME_LIMIT=30
EXP_LIMIT=30000000
NO_LIMIT=999999999

QUICK_TEST=${1:-false}

# Run synthesizer with model and heuristics. Using an expression limit but not
# time limit should be consistent across machines.

java -cp ".:lib/*" com.googleresearch.bustle.Synthesizer \
--quick_test=${QUICK_TEST} \
--sygus_benchmarks=false \
--model_reweighting=true \
--heuristic_reweighting=true \
--premise_selection=false \
--time_limit=${NO_LIMIT} \
--max_expressions=${EXP_LIMIT} \
--model_directory=${BUSTLE_MODEL} \
--output_file=${RESULTS_DIR}/new_tasks_exp_limit.json

java -cp ".:lib/*" com.googleresearch.bustle.Synthesizer \
--quick_test=${QUICK_TEST} \
--sygus_benchmarks=true \
--sygus_benchmarks_directory=${DIR}/sygus_benchmarks \
--model_reweighting=true \
--heuristic_reweighting=true \
--premise_selection=false \
--time_limit=${NO_LIMIT} \
--max_expressions=${EXP_LIMIT} \
--model_directory=${BUSTLE_MODEL} \
--output_file=${RESULTS_DIR}/sygus_tasks_exp_limit.json
