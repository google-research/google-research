# Copyright 2019 The Google Research Authors.
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

pip install -r graph_embedding/watch_your_step/requirements.txt

curl http://sami.haija.org/graph/datasets.tgz > datasets.tgz
tar zxvf datasets.tgz
export DATA_DIR=datasets

# note -- these are not the recommended settings for this dataset.  This is just so the open-source tests will finish quickly.
python -m graph_embedding.watch_your_step.graph_attention_learning --dataset_dir ${DATA_DIR}/wiki-vote --transition_powers 2 --max_number_of_steps 10
