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



set -e
set -x

bazel build -c opt main:colored_c_nn_random_grids_index_main

bazel-bin/main/colored_c_nn_random_grids_index_main \
  --index_descriptor_files testdata/mapillar_10.txt \
  --query_descriptor_files testdata/mapillar_5.txt \
  --pairs_file /tmp/pairs_10_5.txt
