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

python3 -m venv graphqa
source graphqa/bin/activate

pip3 install -r graphqa/requirements.txt

# Fill in appropriate output path
OUTPUT_PATH="~/graphqa/graphs"

echo "The output path is set to: $OUTPUT_PATH"

for algorithm in "er" "ba" "sbm" "sfn" "complete" "star" "path"
do
  echo "Generating test examples for $algorithm"
  python3 -m graphqa.graph_generator \
                    --algorithm=$algorithm \
                    --number_of_graphs=500 \
                    --split=test \
                    --output_path=$OUTPUT_PATH
done
