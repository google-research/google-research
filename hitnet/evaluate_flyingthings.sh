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

# Location of stereo datasets.
DATASETS_LOCATION="/tmp/datasets/"
MODEL_PATH="/tmp/hitnet/models"

MODEL_NAME="flyingthings_finalpass_xl.pb"
LEFT_PATTERN="flying/finalpass/TEST/**/**/left/*.png"
RIGHT_PATTERN="flying/finalpass/TEST/**/**/right/*.png"
# MODEL_NAME="flyingthings_cleanpass_xl.pb"
# LEFT_PATTERN="flying/cleanpass/TEST/**/**/left/*.png"
# RIGHT_PATTERN="flying/cleanpass/TEST/**/**/right/*.png"
# GT is the same for cleanpass and finalpass
GT_LEFT_PATTERN="flying/TEST/**/**/left/*.pfm"
GT_RIGHT_PATTERN="flying/TEST/**/**/right/*.pfm"

#!/bin/bash
set -e
set -x

python3 -m venv pyenv_tf2
source pyenv_tf2/bin/activate

pip3 install -r requirements.txt

mkdir -p $MODEL_PATH
wget -P $MODEL_PATH -N https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/$MODEL_NAME

python -m predict \
  --data_pattern=$DATASETS_LOCATION \
  --model_path=$MODEL_PATH/$MODEL_NAME \
  --png_disparity_factor=128 \
  --iml_pattern=$LEFT_PATTERN \
  --imr_pattern=$RIGHT_PATTERN \
  --gtl_pattern=$GT_LEFT_PATTERN \
  --gtr_pattern=$GT_RIGHT_PATTERN \
  --input_channels=3 \
  --predict_right=true \
  --save_png=false \
  --save_pfm=false \
  --evaluate=true \
  --max_test_number=10000 \  

