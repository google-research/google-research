# Copyright 2024 The Google Research Authors.
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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r bangbang_qaoa/requirements.txt
python -m bangbang_qaoa.search_bangbang_qaoa_protocols \
  --num_literals=10 \
  --num_clauses=10 \
  --initialization_method=random \
  --num_chunks=100 \
  --total_time=3 \
  --num_samples=1
