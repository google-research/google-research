# Copyright 2021 The Google Research Authors.
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

pip install -r quantum_sample_learning/requirements.txt

# Smoke test of the model.
# For full training, please use the flags in README.md.
python3 -m quantum_sample_learning.run_lm \
--use_theoretical_distribution=False \
--probabilities_path=quantum_sample_learning/data/q12c0.txt \
--experimental_bitstrings_path=quantum_sample_learning/data/experimental_samples_q12c0d14_test.txt \
--num_qubits=12 \
--epochs=2 \
--rnn_units=32 \
--eval_samples=500 \
--training_eval_samples=4000
