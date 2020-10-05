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

# For existing Ubuntu systems, I find it easier to use conda since it ships with the correct CUDA version.
# Before running this script, be sure to do:
#   conda create -n py3-ravens python=3.7
#   conda activate py3-ravens
# Then run this script:: ./install_python_ubuntu.sh
# We need to upgrade tensorflow-addons so that it doesn't throw the error:
#   https://github.com/tensorflow/addons/issues/1132
# Note: this uses TF 2.2 whereas the CoRL submission used TF 2.1.

echo "Installing Python libraries..."
conda install ipython
conda install tensorflow-gpu==2.2

pip install numpy==1.17.4
pip install pybullet==2.8.4
pip install packaging==19.2
pip install matplotlib==3.1.1
pip install opencv-python==4.1.2.30
pip install meshcat==0.0.18
pip install transformations==2020.1.1

pip install scikit-image==0.17.2
pip install circle-fit==0.1.3

pip install tensorflow-addons==0.11.1
pip install tensorflow_hub==0.8.0

pip install -e .
