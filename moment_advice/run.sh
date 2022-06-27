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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

echo "1" > moment_advice/moment_advice_tests_advice.txt
echo "1" > moment_advice/moment_advice_tests_dataset.txt

for i in {2..10000}; do
  echo "$i" >> moment_advice/moment_advice_tests_advice.txt
done

for i in {2..10000}; do
  echo "$i" >> moment_advice/moment_advice_tests_dataset.txt
done

for i in {1..1000}; do
  echo "$i" >> moment_advice/moment_advice_tests_dataset.txt
done

for i in {1..1000}; do
  echo "$i" >> moment_advice/moment_advice_tests_dataset.txt
done

pip install -r moment_advice/requirements.txt
python -m moment_advice.moment_advice --dataset_type="unweighted_elements" \
--train_path="moment_advice/moment_advice_tests_advice.txt" \
--test_path="moment_advice/moment_advice_tests_dataset.txt"

rm moment_advice/moment_advice_tests_advice.txt
rm moment_advice/moment_advice_tests_dataset.txt
