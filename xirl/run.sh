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

pip install --upgrade pip

pip install -r xirl/requirements.txt

python -m xirl.pretrain --alsologtostderr \
    --experiment_name="test" \
    --config="xirl/configs/pretraining/default.py" \
    --config.DATA.ROOT="xirl/tests/data/processed/" \
    --config.OPTIM.TRAIN_MAX_ITERS=5 \
    --config.EVAL.EVAL_FREQUENCY=3 \
    --config.FRAME_SAMPLER.NUM_FRAMES_PER_SEQUENCE=3
