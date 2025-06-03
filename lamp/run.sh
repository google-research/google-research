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



# Example commands for LAMP code, see README.md for compiling the binary and
# obtaining the Brightkite dataset.

# Number of iterations, lamp, and ngram order parameters are reduced compared to
# the paper's plots to make it run quicker. It's only meant to illustrate the
# usage of the commands as the LAMP model is not properly optimized.
time ./lamp --max_outer_iter=2 --grad_max_weight_iter=10 --grad_max_transitions_iter=10 --max_train_time=2010 --dataset=brightkite --plot_file brightkite-performance.tsv --min_location_count 10 --max_lamp_order 3 --max_ngram_order 3

# PDF figures are created in the ../figs directory.
./plotall.sh brightkite-performance.tsv brightkite
