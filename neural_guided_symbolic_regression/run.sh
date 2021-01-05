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

virtualenv -p python .
source ./bin/activate

pip install -r neural_guided_symbolic_regression/requirements.txt
python -m neural_guided_symbolic_regression.models.run_partial_sequence_model \
--hparams="$(pwd)/neural_guided_symbolic_regression/models/config/example_network.json" \
--is_chief \
--model_dir=/tmp/neural_guided_symbolic_regression/example_run
