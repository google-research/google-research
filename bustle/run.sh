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



# This script is just a simple test to make sure the individual scripts work.

set -e

# Go to the directory where this script is.
dir=$(dirname "$(readlink -f "$0")")
cd "${dir}"

bash compile.sh
bash run_tests.sh
bash generate_data.sh 5
bash train.sh 10
bash download_sygus_benchmarks.sh
bash run_synthesis.sh true
bash clean.sh
