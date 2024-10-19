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

# Location of stereo datasets.
DATASETS_LOCATION="/tmp/datasets/"
MODEL_PATH="/tmp/hitnet/models"

MODEL_NAME="eth3d.pb"
DATA_PATTERN="$DATASETS_LOCATION/eth3d/train/"
# DATA_PATTERN="$DATASETS_LOCATION/eth3d/test/"
LEFT_PATTERN="**/im0.png"
RIGHT_PATTERN="**/im1.png"
GT_LEFT_PATTERN="**/disp0GT.pfm"

#!/bin/bash
set -e
set -x

python3 -m venv pyenv_tf2
source pyenv_tf2/bin/activate

pip3 install -r requirements.txt

mkdir -p $MODEL_PATH
wget -P $MODEL_PATH -N https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/$MODEL_NAME

python -m predict.py \
  --data_pattern=$DATA_PATTERN \
  --model_path=$MODEL_PATH/$MODEL_NAME \
  --png_disparity_factor=1024 \
  --iml_pattern=$LEFT_PATTERN \
  --imr_pattern=$RIGHT_PATTERN \
  --gtl_pattern=$GT_LEFT_PATTERN \
  --input_channels=1 \
  --predict_right=false \
  --save_png=true \
  --save_pfm=true \
  --evaluate=false \
  --max_test_number=10000 \
