# Copyright 2020 The Google Research Authors.
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

virtualenv -p python3 cnn_quantization_py
source cnn_quantization_py/bin/activate

pip install tf-nightly
pip install -r cnn_quantization/requirements.txt
python -m cnn_quantization.tf_cnn_benchmarks.tf_cnn_benchmarks --data_format=NHWC --batch_size=2 --num_batches=5\
          --model=resnet20_v2 --data_name=cifar10 \
          --use_relu_x=true --quant_act=true --quant_act_bits=4
