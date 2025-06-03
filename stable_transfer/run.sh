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


# Example run for the transferability experiment.
# NOTE: run from parent directory
# ./stable_transfer/run.sh

# conda create --name stable_transfer python=3.9
# conda activate stable_transfer
# pip install -r requirements.txt

echo "Experiment: compute LEEP on CIFAR10 using ResNet50 (ImageNet pretraining)"
echo "1) Download and create the CIFAR10 dataset (using tfds);"
echo "2) Evaluate the ResNet model on all images;"
echo "3) Compute the LEEP score."
echo "This might be slow when run on CPU (consider installing on CUDA & CUDNN)."
python -m stable_transfer.transferability.main

echo "The same experiment. But now the results should be loaded from disk."

python -m stable_transfer.transferability.main
