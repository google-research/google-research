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

set -e

echo "Compile Generative Decision Trees project"
javac -d compiled -classpath src src/*.java

echo "Prints the help"
java -classpath compiled Wrapper --help

echo "Download a copy of the Iris dataset"
wget https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/iris.csv -O iris.csv

echo "Train and sample a new generator"
mkdir -p working_dir
java -classpath compiled Wrapper \
  --dataset=iris.csv \
  --work_dir=working_dir \
  --num_samples=1000 \
  --output_samples=working_dir/generated.csv \
  --output_stats=working_dir/statistics.stats

echo "Display some of the generated samples"
head working_dir/generated.csv
