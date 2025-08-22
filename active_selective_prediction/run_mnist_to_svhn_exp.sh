#!/bin/bash
# Copyright 2025 The Google Research Authors.
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
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r active_selective_prediction/requirements.txt
TF_CPP_MIN_LOG_LEVEL=2 python -m active_selective_prediction.train --gpu 0 --dataset color_mnist
TF_CPP_MIN_LOG_LEVEL=2 python -m active_selective_prediction.eval_model --gpu 0 --source-dataset color_mnist --model-path ./checkpoints/standard_supervised/color_mnist
TF_CPP_MIN_LOG_LEVEL=2 python -m active_selective_prediction.eval_pipeline --gpu 0 --source-dataset color_mnist --method sr --method-config-file ./active_selective_prediction/configs/sr.json
TF_CPP_MIN_LOG_LEVEL=2 python -m active_selective_prediction.eval_pipeline --gpu 0 --source-dataset color_mnist --method de --method-config-file ./active_selective_prediction/configs/de.json
TF_CPP_MIN_LOG_LEVEL=2 python -m active_selective_prediction.eval_pipeline --gpu 0 --source-dataset color_mnist --method aspest --method-config-file ./active_selective_prediction/configs/aspest.json
