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


echo "Replicating all results in paper"
echo "Note this code takes a week to run on a single machine"
echo "ADMM is particularly slow on nug20 as the factorization has a lot of fill"
echo "To quickly replicate a single result use run_configuration.jl"

test_problems_folder="${1:?missing argument test_problems_folder}"
results_folder="${2:?missing argument results_folder}"
scripts_folder="$( dirname -- "$0" )"

echo "LP test problem folder = ${test_problems_folder}"
echo "Results problem folder = ${results_folder}"
echo "scripts folder = ${scripts_folder}"

for method_name in extragradient PDHG
do
  echo "Create bilinear figures for ${method_name}"
  mkdir -p "${results_folder}/bilinear-example/${method_name}"
  julia --project \
    "${scripts_folder}/run_and_plot_bilinear_example.jl" \
    "${results_folder}/bilinear-example/${method_name}" "${method_name}" \
    > "${results_folder}/bilinear-example/${method_name}/log.txt"
done

for method_name in ADMM extragradient PDHG
do
  echo "Running ${method_name} to produce data for Figures"
  echo "This could take a couple of days to run"
  julia --project "${scripts_folder}/run_problems.jl" \
    "${test_problems_folder}" \
    "${results_folder}/${method_name}/${problem_name}" \
    "${method_name}" \
    all

  echo "Producing figures"
  julia --project plot_results.jl \
    "${test_problems_folder}" \
    "{$results_folder}/${method_name}/${problem_name}"
done

echo "Compute the column of the Tables for no restarts."
echo "This is done seperately with a larger iteration limit"
echo "to save compute."

echo "Run ADMM with no restarts"
for problem_name in qap10 qap15 nug08-3rd nug20
do
  configuration_folder="${results_folder}/ADMM/${problem_name}"
  mkdir -p "${configuration_folder}"
  julia --project "${scripts_folder}/run_configuration.jl" \
    "${"test_problems_folder}" \
    "${configuration_folder}/no_restarts_1e-6.csv" \
    ADMM \
    "${problem_name}" \
    no_restarts \
    0 \
    no \
    200000 \
    "1e-6" \
    > "${configuration_folder}/no_restarts_1e-6_log.txt"
done

for method_name in extragradient PDHG
do
  echo "Run ${method_name} with no restarts"
  for problem_name in qap10 qap15 nug08-3rd nug20
  do
    configuration_folder="${results_folder}/${method_name}/${problem_name}"
    mkdir -p "${configuration_folder}"
    julia --project "${scripts_folder}/run_configuration.jl" \
      ${test_problems_folder} \
      "${configuration_folder}/no_restarts_1e-6.csv" \
      "${method_name}" \
      "${problem_name}" \
      no_restarts \
      0 \
      no \
      500000 \
      "1e-6" \
      > "${configuration_folder}/no_restarts_1e-6_log.txt"
  done
done

