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
# export DATA_DIR=[path/to/your/data]

# Dog-vs-cat dataset
# 1. Download cats and dogs dataset
#   https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip
# 2. Run the following script to create numpy array
python -m deep_representation_one_class.script.prepare_dogvscat.py

# CelebA dataset
# 1. Run the following script to download and create numpy array
python -m deep_representation_one_class.script.prepare_celeba.py
