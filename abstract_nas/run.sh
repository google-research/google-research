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



# Install our requirements.
pip install -r abstract_nas/requirements.txt

# Get the Big Vision codebase at a specific commit in case APIs have shifted.
git clone --branch=main https://github.com/google-research/big_vision
cd big_vision
git checkout 6c376d6f621fe93627ee697674abd202392d8faa
cd ..
pip install -r big_vision/big_vision/requirements.txt

# Prepare a local copy of the data set.
PYTHONPATH=big_vision:$PYTHONPATH python -m big_vision.tools.download_tfds_datasets cifar10

# Run a toy experiment to evolve a 2-layer CNN on CIFAR-10 for 50 generations.
PYTHONPATH=big_vision:$PYTHONPATH python -m abstract_nas.evolution.main \
  --max_num_trials=50 \
  --study_name="abstract_nas_demo" \
  --results_dir="." \
  --nocheckpoints \
  --config="abstract_nas/evolution/config.py:cnn.cifar10"
