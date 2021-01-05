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

pip install tensorflow
pip install -r protein_lm/requirements.txt
python -m protein_lm.generate_data --input_dir $PWD/protein_lm/testdata --output_dir /tmp/protein_lm/data --alsologtostderr
python -m protein_lm.train --gin_bindings "experiment.data_dir = '/tmp/protein_lm/data'" --work_dir /tmp/protein_lm/model --alsologtostderr
