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
set -uexo pipefail

echo "Installing Python libraries..."
python3 -m pip install --user numpy==1.17.4
python3 -m pip install --user pybullet==2.8.4
python3 -m pip install --user packaging==19.2
python3 -m pip install --user matplotlib==3.1.1
python3 -m pip install --user opencv-python==4.1.2.30
python3 -m pip install --user meshcat==0.0.18
python3 -m pip install --user transformations==2020.1.1

echo "Installing extra libraries for supporting deformables..."
python3 -m pip install --user scikit-image==0.17.2
python3 -m pip install --user circle-fit==0.1.3

echo "Installing Tensorflow..."
python3 -m pip install --user tensorflow==2.1.0
python3 -m pip install --user tensorflow-addons==0.8.3
python3 -m pip install --user tensorflow_hub==0.8.0

echo "Installing this package ..."
python3 -m pip install --user -e .
