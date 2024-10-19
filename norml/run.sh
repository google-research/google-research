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

pip install -r norml/requirements.txt

wget -P norml/example_checkpoints/move_point_rotate_sparse/norml/ https://storage.googleapis.com/gresearch/norml/example_checkpoints/move_point_rotate_sparse/norml/all_weights.ckpt-991.data-00000-of-00001
wget -P norml/example_checkpoints/move_point_rotate_sparse/norml/ https://storage.googleapis.com/gresearch/norml/example_checkpoints/move_point_rotate_sparse/norml/all_weights.ckpt-991.index
wget -P norml/example_checkpoints/move_point_rotate_sparse/norml/ https://storage.googleapis.com/gresearch/norml/example_checkpoints/move_point_rotate_sparse/norml/all_weights.ckpt-991.meta
wget -P norml/example_checkpoints/move_point_rotate_sparse/norml/ https://storage.googleapis.com/gresearch/norml/example_checkpoints/move_point_rotate_sparse/norml/config.yaml
python -m norml.eval_maml --model_dir example_checkpoints/move_point_rotate_sparse/norml/all_weights.ckpt-991 --output_dir /usr/local/google/home/yxyang/temp --render=False --num_finetune_steps 1 --test_task_index 0 --eval_finetune=True
